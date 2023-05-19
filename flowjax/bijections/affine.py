"""Affine bijections."""

from typing import Callable
import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike
import warnings

from flowjax.bijections.bijection import Bijection


class Affine(Bijection):
    """Elementwise affine transformation ``y = a*x + b``. loc and scale should broadcast
    to the desired shape of the bijection.
    """

    loc: Array
    log_scale: Array

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """
        Args:
            loc (ArrayLike): Location parameter. Defaults to 0.
            scale (ArrayLike): Scale parameter. Defaults to 1.
        """
        loc, scale = [jnp.asarray(a, dtype=float) for a in (loc, scale)]
        self.shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        self.cond_shape = None

        self.loc = jnp.broadcast_to(loc, self.shape)
        self.log_scale = jnp.broadcast_to(jnp.log(scale), self.shape)

    def transform(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return x * self.scale + self.loc

    def transform_and_log_det(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return x * self.scale + self.loc, self.log_scale.sum()

    def inverse(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        return (y - self.loc) / self.scale

    def inverse_and_log_det(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        return (y - self.loc) / self.scale, -self.log_scale.sum()

    @property
    def scale(self):
        """The scale parameter of the affine transformation."""
        return jnp.exp(self.log_scale)


class TriangularAffine(Bijection):
    """Transformation of the form ``Ax + b``, where ``A`` is a lower or upper triangular matrix."""

    loc: Array
    diag_idxs: Array
    tri_mask: Array
    lower: bool
    weight_log_scale: Array | None
    _arr: Array
    _log_diag: Array

    def __init__(
        self,
        loc: Array,
        arr: Array,
        lower: bool = True,
        weight_normalisation: bool = False,
    ):
        """
        Args:
            loc (Array): Location parameter.
            arr (Array): Triangular matrix.
            lower (bool): Whether the mask should select the lower or upper
                triangular matrix (other elements ignored). Defaults to True.
            weight_log_scale (Array | None): If provided, carry out weight
                normalisation.
        """
        if (arr.ndim != 2) or (arr.shape[0] != arr.shape[1]):
            raise ValueError("arr must be a square, 2-dimensional matrix.")
        checkify.check(
            jnp.all(jnp.diag(arr) > 0),
            "arr diagonal entries must be greater than 0",
        )
        dim = arr.shape[0]
        self.diag_idxs = jnp.diag_indices(dim)
        tri_mask = jnp.tril(jnp.ones((dim, dim), dtype=jnp.int32), k=-1)
        self.tri_mask = tri_mask if lower else tri_mask.T
        self.lower = lower

        # inexact arrays
        self.loc = jnp.broadcast_to(loc, (dim,))
        self._arr = arr
        self._log_diag = jnp.log(jnp.diag(arr))
        self.weight_log_scale = jnp.zeros((dim, 1)) if weight_normalisation else None

        self.shape = (dim,)
        self.cond_shape = None

    @property
    def arr(self):
        """Get triangular array, (applies masking and min_diag constraint)."""
        diag = jnp.exp(self._log_diag)
        off_diag = self.tri_mask * self._arr
        arr = off_diag.at[self.diag_idxs].set(diag)

        if self.weight_log_scale is not None:
            norms = jnp.linalg.norm(arr, axis=1, keepdims=True)
            arr = jnp.exp(self.weight_log_scale) * arr / norms

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


class AdditiveLinearCondition(Bijection):
    """Deprecated as of v9.1.0. Use ``AdditiveCondition`` bijection instead.

    Carries out ``y = x + W @ condition``, as the forward transformation and
    ``x = y - W @ condition`` as the inverse.
    """

    W: Array

    def __init__(self, arr: Array):
        """
        Args:
            arr (Array): Array (``W`` in the description.)
        """
        warnings.warn(
            "AdditiveLinearCondition is deprecated in favour of the more general "
            "AdditiveCondition as of v9.1.0."
        )
        self.W = arr
        self.shape = (arr.shape[-2],)
        self.cond_shape = (arr.shape[-1],)

    def transform(self, x, condition=None):
        x, condition = self._argcheck_and_cast(x, condition)
        return x + self.W @ condition

    def transform_and_log_det(self, x, condition=None):
        x, condition = self._argcheck_and_cast(x, condition)
        return self.transform(x, condition), jnp.array(0)

    def inverse(self, y, condition=None):
        y, condition = self._argcheck_and_cast(y, condition)
        return y - self.W @ condition

    def inverse_and_log_det(self, y, condition=None):
        y, condition = self._argcheck_and_cast(y, condition)
        return self.inverse(y, condition), jnp.array(0)
