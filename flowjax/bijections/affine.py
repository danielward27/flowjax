"""Affine bijections."""
from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.softplus import SoftPlus
from flowjax.utils import arraylike_to_array


class Affine(AbstractBijection):
    """Elementwise affine transformation ``y = a*x + b``.

    ``loc`` and ``scale`` should broadcast to the desired shape of the bijection.

    Args:
        loc: Location parameter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
        positivity_constraint: Bijection with shape matching the Affine bijection, that
            maps the scale parameter from an unbounded domain to the positive domain.
            Defaults to :class:`~flowjax.bijections.SoftPlus`.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    _scale: Array
    positivity_constraint: AbstractBijection

    def __init__(
        self,
        loc: ArrayLike = 0,
        scale: ArrayLike = 1,
        positivity_constraint: AbstractBijection | None = None,
    ):
        loc, scale = (arraylike_to_array(a, dtype=float) for a in (loc, scale))
        self.shape = jnp.broadcast_shapes(loc.shape, scale.shape)
        self.loc = jnp.broadcast_to(loc, self.shape)

        if positivity_constraint is None:
            positivity_constraint = SoftPlus(self.shape)

        self.positivity_constraint = positivity_constraint
        self._scale = positivity_constraint.inverse(jnp.broadcast_to(scale, self.shape))

    def transform(self, x, condition=None):
        return x * self.scale + self.loc

    def transform_and_log_det(self, x, condition=None):
        scale = self.scale
        return x * scale + self.loc, jnp.log(scale).sum()

    def inverse(self, y, condition=None):
        return (y - self.loc) / self.scale

    def inverse_and_log_det(self, y, condition=None):
        scale = self.scale
        return (y - self.loc) / scale, -jnp.log(scale).sum()

    @property
    def scale(self):
        """The scale parameter of the affine transformation."""
        return self.positivity_constraint.transform(self._scale)


class TriangularAffine(AbstractBijection):
    r"""A triangular affine transformation.

    Transformation has the form :math:`Ax + b`, where :math:`A` is a lower or upper
    triangular matrix, and :math:`b` is the bias vector.

    Args:
        loc: Location parameter. If this is scalar, it is broadcast to the dimension
            inferred from arr.
        arr: Triangular matrix.
        lower: Whether the mask should select the lower or upper
            triangular matrix (other elements ignored). Defaults to True (lower).
        weight_normalisation: If true, carry out weight normalisation.
        positivity_constraint: Bijection with shape matching the dimension of the
            triangular affine bijection, that maps the diagonal entries of the array
            from an unbounded domain to the positive domain. Also used for weight
            normalisation parameters, if used. Defaults to SoftPlus.
    """
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    diag_idxs: Array
    tri_mask: Array
    lower: bool
    positivity_constraint: AbstractBijection
    _arr: Array
    _diag: Array
    _weight_scale: Array | None

    def __init__(
        self,
        loc: ArrayLike,
        arr: ArrayLike,
        *,
        lower: bool = True,
        weight_normalisation: bool = False,
        positivity_constraint: AbstractBijection | None = None,
    ):
        loc, arr = (arraylike_to_array(a, dtype=float) for a in (loc, arr))
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
        """Get the triangular array.

        Applies masking, constrains the diagonal to be positive and (possibly)
        applies weight normalisation.
        """
        diag = self.positivity_constraint.transform(self._diag)
        off_diag = self.tri_mask * self._arr
        arr = off_diag.at[self.diag_idxs].set(diag)

        if self._weight_scale is not None:
            norms = jnp.linalg.norm(arr, axis=1, keepdims=True)
            scale = self.positivity_constraint.transform(self._weight_scale)[:, None]
            arr = scale * arr / norms

        return arr

    def transform(self, x, condition=None):
        return self.arr @ x + self.loc

    def transform_and_log_det(self, x, condition=None):
        arr = self.arr
        return arr @ x + self.loc, jnp.log(jnp.diag(arr)).sum()

    def inverse(self, y, condition=None):
        return solve_triangular(self.arr, y - self.loc, lower=self.lower)

    def inverse_and_log_det(self, y, condition=None):
        arr = self.arr
        x = solve_triangular(arr, y - self.loc, lower=self.lower)
        return x, -jnp.log(jnp.diag(arr)).sum()


class AdditiveCondition(AbstractBijection):
    """Given a callable ``f``, carries out the transformation ``y = x + f(condition)``.

    If used to transform a distribution, this allows the "location" to be changed as a
    function of the conditioning variables. Note that the callable can be a callable
    module with trainable parameters.

    Args:
        module: A callable (e.g. a function or callable module) that maps array with
            shape cond_shape, to a shape that is broadcastable with the shape of the
            bijection.
        shape: The shape of the bijection.
        cond_shape: The condition shape of the bijection.

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

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    module: Callable[[ArrayLike], ArrayLike]

    def __init__(
        self,
        module: Callable[[ArrayLike], ArrayLike],
        shape: tuple[int, ...],
        cond_shape: tuple[int, ...],
    ):
        self.module = module
        self.shape = shape
        self.cond_shape = cond_shape

    def transform(self, x, condition=None):
        return x + self.module(condition)

    def transform_and_log_det(self, x, condition=None):
        return self.transform(x, condition), jnp.array(0)

    def inverse(self, y, condition=None):
        return y - self.module(condition)

    def inverse_and_log_det(self, y, condition=None):
        return self.inverse(y, condition), jnp.array(0)
