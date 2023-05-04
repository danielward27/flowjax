"""Affine bijections."""

import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

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
            loc (int): Location parameter. Defaults to 0.
            scale (int): Scale parameter. Defaults to 1.
        """
        loc, scale = [jnp.asarray(a, dtype=jnp.float32) for a in (loc, scale)]
        self.shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        self.cond_shape = None

        self.loc = jnp.broadcast_to(loc, self.shape)
        self.log_scale = jnp.broadcast_to(jnp.log(scale), self.shape)

    def transform(self, x, condition=None):
        self._argcheck(x)
        return x * self.scale + self.loc

    def transform_and_log_det(self, x, condition=None):
        self._argcheck(x)
        return x * self.scale + self.loc, self.log_scale.sum()

    def inverse(self, y, condition=None):
        self._argcheck(y)
        return (y - self.loc) / self.scale

    def inverse_and_log_det(self, y, condition=None):
        self._argcheck(y)
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
        self._argcheck(x)
        return self.arr @ x + self.loc

    def transform_and_log_det(self, x, condition=None):
        self._argcheck(x)
        arr = self.arr
        return arr @ x + self.loc, jnp.log(jnp.diag(arr)).sum()

    def inverse(self, y, condition=None):
        self._argcheck(y)
        return solve_triangular(self.arr, y - self.loc, lower=self.lower)

    def inverse_and_log_det(self, y, condition=None):
        self._argcheck(y)
        arr = self.arr
        x = solve_triangular(arr, y - self.loc, lower=self.lower)
        return x, -jnp.log(jnp.diag(arr)).sum()


class AdditiveLinearCondition(Bijection):
    """Carries out ``y = x + W @ condition``, as the forward transformation and
    ``x = y - W @ condition`` as the inverse.
    """

    W: Array

    def __init__(self, arr: Array):
        """
        Args:
            arr (Array): Array (``W`` in the description.)
        """
        self.W = arr
        self.shape = (arr.shape[-2],)
        self.cond_shape = (arr.shape[-1],)

    def transform(self, x, condition=None):
        self._argcheck(x, condition)
        return x + self.W @ condition

    def transform_and_log_det(self, x, condition=None):
        self._argcheck(x, condition)
        return self.transform(x, condition), jnp.array(0)

    def inverse(self, y, condition=None):
        self._argcheck(y, condition)
        return y - self.W @ condition

    def inverse_and_log_det(self, y, condition=None):
        self._argcheck(y, condition)
        return self.inverse(y, condition), jnp.array(0)
