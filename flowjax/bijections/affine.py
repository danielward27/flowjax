from typing import Optional

import jax.numpy as jnp
from jax.experimental import checkify
from jax.scipy.linalg import solve_triangular

from flowjax.bijections import Bijection
from flowjax.utils import Array, broadcast_arrays_1d


class Affine(Bijection):
    """Elementwise affine transformation. Condition is ignored."""
    loc: Array
    log_scale: Array
    dim: int

    def __init__(self, loc: Array, scale: Array = 1.0):
        """``loc`` and ``scale`` should broadast to the dimension of the transformation.

        Args:
            loc (Array): Location parameter vector.
            scale (Array): Positive scale parameter vector.
        """
        loc, scale = broadcast_arrays_1d(loc, scale)
        self.loc = loc
        self.log_scale = jnp.log(scale)
        self.dim = loc.shape[0]
        self.cond_dim = 0

    def transform(self, x, condition=None):
        return x * self.scale + self.loc

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        self._argcheck(x)
        return x * self.scale + self.loc, self.log_scale.sum()

    def inverse(self, y, condition=None):
        return (y - self.loc) / self.scale

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        self._argcheck(y)
        return (y - self.loc) / self.scale, -self.log_scale.sum()

    @property
    def scale(self):
        return jnp.exp(self.log_scale)

    def _argcheck(self, x: Array):
        if x.shape != (self.dim,):
            raise ValueError(f"Expected shape {(self.dim, )}, got {x.shape}.")


class TriangularAffine(Bijection):
    """Transformation of the form ``Ax + b``, where ``A`` is a lower or upper triangular matrix."""
    loc: Array
    dim: int
    cond_dim: int
    diag_idxs: Array
    tri_mask: Array
    lower: bool
    min_diag: float
    weight_log_scale: Optional[Array]
    _arr: Array
    _log_diag: Array

    def __init__(
        self,
        loc: Array,
        arr: Array,
        lower: bool = True,
        min_diag: float = 1e-6,
        weight_normalisation: bool = False,
    ):
        """
        Args:
            loc (Array): Translation.
            arr (Array): Triangular matrix.
            lower (bool, optional): Whether the mask should select the lower or upper triangular matrix (other elements ignored). Defaults to True.
            min_diag (float, optional): Minimum value on the diagonal, to ensure invertibility. Defaults to 1e-6.
            weight_log_scale (Optional[Array], optional): If provided, carry out weight normalisation, initialising log scales to the zero vector.
        """

        if (arr.ndim != 2) or (arr.shape[0] != arr.shape[1]):
            ValueError("arr must be a square, 2-dimensional matrix.")
        checkify.check(
            jnp.all(jnp.diag(arr) > min_diag),
            "arr diagonal entries must be greater than min_diag",
        )

        self.dim = arr.shape[0]
        self.cond_dim = 0
        self.diag_idxs = jnp.diag_indices(self.dim)
        tri_mask = jnp.tril(jnp.ones((self.dim, self.dim), dtype=jnp.int32), k=-1)
        self.tri_mask = tri_mask if lower else tri_mask.T
        self.min_diag = min_diag
        self.lower = lower

        # inexact arrays
        self.loc = loc
        self._arr = arr
        self._log_diag = jnp.log(jnp.diag(arr) - min_diag)
        self.weight_log_scale = (
            jnp.zeros((self.dim, 1)) if weight_normalisation else None
        )

    @property
    def arr(self):
        "Get triangular array, (applies masking and min_diag constraint)."
        diag = jnp.exp(self._log_diag) + self.min_diag
        off_diag = self.tri_mask * self._arr
        arr = off_diag.at[self.diag_idxs].set(diag)

        if self.weight_log_scale is not None:
            norms = jnp.linalg.norm(arr, axis=1, keepdims=True)
            arr = jnp.exp(self.weight_log_scale) * arr / norms

        return arr

    def transform(self, x, condition=None):
        return self.arr @ x + self.loc

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        a = self.arr
        return a @ x + self.loc, jnp.log(jnp.diag(a)).sum()

    def inverse(self, y, condition=None):
        return solve_triangular(self.arr, y - self.loc, lower=self.lower)

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        a = self.arr
        return (
            solve_triangular(a, y - self.loc, lower=self.lower),
            -jnp.log(jnp.diag(a)).sum(),
        )


class AdditiveLinearCondition(Bijection):
    """Carries out ``y = x + W @ condition``, as the forward transformation and
    ``x = y - W @ condition`` as the inverse."""
    dim: int
    cond_dim: int
    W: Array

    def __init__(self, arr: Array):
        """
        Args:
            arr (Array): Array (``W`` in the description.)
        """
        self.dim = arr.shape[0]
        self.cond_dim = arr.shape[1]
        self.W = arr

    def transform(self, x, condition=None):
        return x + self.W @ condition

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        return self.transform(x, condition), jnp.array(0)

    def inverse(self, y, condition=None):
        return y - self.W @ condition

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        return self.inverse(y, condition), jnp.array(0)
