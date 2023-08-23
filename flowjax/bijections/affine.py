"""Affine bijections."""

from typing import Callable

import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from flowjax.bijections.bijection import Bijection
from flowjax.bijections.softplus import SoftPlus
from flowjax.utils import arraylike_to_array


class Affine(Bijection):
    """Elementwise affine transformation ``y = a*x + b``. loc and scale should broadcast
    to the desired shape of the bijection.
    """

    loc: Array
    _scale: Array
    positivity_constraint: Bijection

    def __init__(
        self,
        loc: ArrayLike = 0,
        scale: ArrayLike = 1,
        positivity_constraint: Bijection | None = None,
    ):
        """
        Args:
            loc (ArrayLike): Location parameter. Defaults to 0.
            scale (ArrayLike): Scale parameter. Defaults to 1.
            postivity_constraint (Bijection): Bijection with shape matching the Affine
                bijection, that maps the scale parameter from an unbounded domain to the
                positive domain. Defaults to SoftPlus.
        """
        loc, scale = [arraylike_to_array(a, dtype=float) for a in (loc, scale)]
        self.shape = jnp.broadcast_shapes(loc.shape, scale.shape)
        self.cond_shape = None

        self.loc = jnp.broadcast_to(loc, self.shape)

        if positivity_constraint is None:
            positivity_constraint = SoftPlus(self.shape)

        self.positivity_constraint = positivity_constraint
        self._scale = positivity_constraint.inverse(jnp.broadcast_to(scale, self.shape))

    def transform(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return x * self.scale + self.loc

    def transform_and_log_det(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        scale = self.scale
        return x * scale + self.loc, jnp.log(scale).sum()

    def inverse(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        return (y - self.loc) / self.scale

    def inverse_and_log_det(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        scale = self.scale
        return (y - self.loc) / scale, -jnp.log(scale).sum()

    @property
    def scale(self):
        """The scale parameter of the affine transformation."""
        return self.positivity_constraint.transform(self._scale)


class TriangularAffine(Bijection):
    r"""Transformation of the form :math:`Ax + b`, where :math:`A` is a lower or upper
    triangular matrix."""

    loc: Array
    diag_idxs: Array
    tri_mask: Array
    lower: bool
    positivity_constraint: Bijection
    _arr: Array
    _diag: Array
    _weight_scale: Array | None

    def __init__(
        self,
        loc: ArrayLike,
        arr: ArrayLike,
        lower: bool = True,
        weight_normalisation: bool = False,
        positivity_constraint: Bijection | None = None,
    ):
        """
        Args:
            loc (ArrayLike): Location parameter.
            arr (ArrayLike): Triangular matrix.
            lower (bool): Whether the mask should select the lower or upper
                triangular matrix (other elements ignored). Defaults to True (lower).
            weight_normalisation (bool): If true, carry out weight normalisation.
            postivity_constraint (Bijection): Bijection with shape matching the
                dimension of the triangular affine bijection, that maps the diagonal
                entries of the array from an unbounded domain to the positive domain.
                Also used for weight normalisation parameters, if used. Defaults to
                SoftPlus.
        """
        loc, arr = [arraylike_to_array(a, dtype=float) for a in (loc, arr)]
        if (arr.ndim != 2) or (arr.shape[0] != arr.shape[1]):
            raise ValueError("arr must be a square, 2-dimensional matrix.")
        checkify.check(
            jnp.all(jnp.diag(arr) > 0),
            "arr diagonal entries must be positive",
        )
        dim = arr.shape[0]
        self.diag_idxs = jnp.diag_indices(dim)
        tri_mask = jnp.tril(jnp.ones((dim, dim), dtype=jnp.int32), k=-1)
        self.tri_mask = tri_mask if lower else tri_mask.T
        self.lower = lower

        self.shape = (dim,)
        self.cond_shape = None

        if positivity_constraint is None:
            positivity_constraint = SoftPlus(self.shape)

        self.positivity_constraint = positivity_constraint
        self._diag = positivity_constraint.inverse(jnp.diag(arr))

        # inexact arrays
        self.loc = jnp.broadcast_to(loc, (dim,))
        self._arr = arr

        if weight_normalisation:
            self._weight_scale = positivity_constraint.inverse(jnp.ones((dim,)))
        else:
            self._weight_scale = None

    @property
    def arr(self):
        """Get triangular array, (applies masking, constrains diagonal and weight
        normalisation)."""
        diag = self.positivity_constraint.transform(self._diag)
        off_diag = self.tri_mask * self._arr
        arr = off_diag.at[self.diag_idxs].set(diag)

        if self._weight_scale is not None:
            norms = jnp.linalg.norm(arr, axis=1, keepdims=True)
            scale = self.positivity_constraint.transform(self._weight_scale)[:, None]
            arr = scale * arr / norms

        return arr

    def transform(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return self.arr @ x + self.loc

    def transform_and_log_det(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        arr = self.arr
        return arr @ x + self.loc, jnp.log(jnp.diag(arr)).sum()

    def inverse(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        return solve_triangular(self.arr, y - self.loc, lower=self.lower)

    def inverse_and_log_det(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        arr = self.arr
        x = solve_triangular(arr, y - self.loc, lower=self.lower)
        return x, -jnp.log(jnp.diag(arr)).sum()


class AdditiveCondition(Bijection):
    """Given a callable ``f``, carries out ``y = x + f(condition)`` as the forward
    transformation and ``x = y - f(condition)`` as the inverse transformation. Note that
    the callable can be a callable module with trainable parameters if desired.

    If used to transform a distribution, this allows the "location" to be changed as a
    function of the conditioning variables.
    """

    module: Callable[[ArrayLike], ArrayLike]

    def __init__(
        self,
        module: Callable[[ArrayLike], ArrayLike],
        shape: tuple[int, ...],
        cond_shape: tuple[int, ...],
    ):
        """
        Args:
            module (Callable[[ArrayLike], ArrayLike]): A callable (e.g. a function or
                callable module) that maps array with shape cond_shape, to a shape
                that is broadcastable with the shape of the bijection.
            shape (tuple[int, ...]): The shape of the bijection.
            cond_shape (tuple[int, ...]): The condition shape of the bijection.

        Example:
            Conditioning using a linear transformation

            .. doctest::

                >>> from flowjax.bijections import AdditiveCondition
                >>> from equinox.nn import Linear
                >>> import jax.numpy as jnp
                >>> import jax.random as jr
                >>> bijection = AdditiveCondition(
                ...     Linear(2, 3, key=jr.PRNGKey(0)), shape=(3,), cond_shape=(2,)
                ...     )
                >>> bijection.transform(jnp.ones(3), condition=jnp.ones(2))
                Array([1.9670618, 0.8156546, 1.7763454], dtype=float32)
        """
        self.module = module
        self.shape = shape
        self.cond_shape = cond_shape

    def transform(self, x, condition=None):
        x, condition = self._argcheck_and_cast(x, condition)
        return x + self.module(condition)  # type: ignore - validated in argcheck

    def transform_and_log_det(self, x, condition=None):
        x, condition = self._argcheck_and_cast(x, condition)
        return self.transform(x, condition), jnp.array(0)

    def inverse(self, y, condition=None):
        y, condition = self._argcheck_and_cast(y, condition)
        return y - self.module(condition)  # type: ignore

    def inverse_and_log_det(self, y, condition=None):
        y, condition = self._argcheck_and_cast(y, condition)
        return self.inverse(y, condition), jnp.array(0)  # type: ignore
