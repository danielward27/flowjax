from typing import Optional

import jax.numpy as jnp
from jax.experimental import checkify
from jax.scipy.linalg import solve_triangular

from flowjax.bijections import Bijection
from flowjax.utils import Array
from jax.experimental import checkify


class Affine(Bijection):
    loc: Array
    log_scale: Array
    
    def __init__(self, loc: Array=0, scale: Array=1):
        """Elementwise affine transformation y = ax + b. loc and scale should broadcast
        to the desired shape of the bijection.

        Args:
            loc (int, optional): Location parameter. Defaults to 0.
            scale (int, optional): Scale parameter. Defaults to 1.
        """
        loc, scale = [jnp.asarray(a, dtype=jnp.float32) for a in (loc, scale)]
        self.shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        self.cond_shape = None

        self.loc = jnp.broadcast_to(loc, self.shape)
        self.log_scale = jnp.broadcast_to(jnp.log(scale), self.shape)

    def transform(self, x, condition=None):
        self._argcheck(x)
        return x * self.scale + self.loc

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        self._argcheck(x)
        return x * self.scale + self.loc, self.log_scale.sum()

    def inverse(self, y, condition=None):
        self._argcheck(y)
        return (y - self.loc) / self.scale

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        self._argcheck(y)
        return (y - self.loc) / self.scale, -self.log_scale.sum()

    @property
    def scale(self):
        return jnp.exp(self.log_scale)


class TriangularAffine(Bijection):
    """Transformation of the form ``Ax + b``, where ``A`` is a lower or upper triangular matrix."""

    loc: Array
    diag_idxs: Array
    tri_mask: Array
    lower: bool
    weight_log_scale: Optional[Array]
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
            lower (bool, optional): Whether the mask should select the lower or upper triangular matrix (other elements ignored). Defaults to True.
            weight_log_scale (Optional[Array], optional): If provided, carry out weight normalisation.
        """

        if (arr.ndim != 2) or (arr.shape[0] != arr.shape[1]):
            ValueError("arr must be a square, 2-dimensional matrix.")
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
        "Get triangular array, (applies masking and min_diag constraint)."
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

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        self._argcheck(x)
        a = self.arr
        return a @ x + self.loc, jnp.log(jnp.diag(a)).sum()

    def inverse(self, y, condition=None):
        self._argcheck(y)
        return solve_triangular(self.arr, y - self.loc, lower=self.lower)

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        self._argcheck(y)
        a = self.arr
        x = solve_triangular(a, y - self.loc, lower=self.lower)
        return x, -jnp.log(jnp.diag(a)).sum()


class AdditiveLinearCondition(Bijection):
    """Carries out ``y = x + W @ condition``, as the forward transformation and
    ``x = y - W @ condition`` as the inverse."""

    W: Array

    def __init__(self, arr: Array):
        """
        Args:
            arr (Array): Array (``W`` in the description.)
        """
        self.W = arr
        super().__init__(shape=(arr.shape[-2],), cond_shape=(arr.shape[-1],))

    def transform(self, x, condition=None):
        self._argcheck(x, condition)
        return x + self.W @ condition

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        self._argcheck(x, condition)
        return self.transform(x, condition), jnp.array(0)

    def inverse(self, y, condition=None):
        self._argcheck(y, condition)
        return y - self.W @ condition

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        self._argcheck(y, condition)
        return self.inverse(y, condition), jnp.array(0)
