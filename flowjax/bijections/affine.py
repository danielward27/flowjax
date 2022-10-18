from flowjax.bijections import Bijection
from flowjax.utils import Array, broadcast_arrays_1d
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

class Affine(Bijection):
    loc: Array
    log_scale: Array
    dim: int
    
    def __init__(self, loc: Array, scale: Array = 1.0):
        """Elementwise affine transformation. Condition is ignored. loc and scale
        should be broadcastable.

        Args:
            loc (Array): Location parameter vector.
            scale (Array): Positive scale parameter vector.
        """
        loc, scale = broadcast_arrays_1d(loc, scale)
        self.loc = loc
        self.log_scale = jnp.log(scale)
        self.dim = loc.shape[0]
        self.cond_dim = 0

    def transform(self, x, condition = None):
        return x * self.scale + self.loc

    def transform_and_log_abs_det_jacobian(self, x, condition = None):
        self._argcheck(x)
        return x * self.scale + self.loc, self.log_scale.sum()

    def inverse(self, y, condition = None):
        return (y - self.loc) / self.scale

    def inverse_and_log_abs_det_jacobian(self, y, condition = None):
        self._argcheck(y)
        return (y - self.loc) / self.scale, -self.log_scale.sum()

    @property
    def scale(self):
        return jnp.exp(self.log_scale)

    def _argcheck(self, x: Array):
        if x.shape != (self.dim, ):
            raise ValueError(f"Expected shape {(self.dim, )}, got {x.shape}.")



class TriangularAffine(Bijection):
    loc: Array
    dim: int
    cond_dim: int
    diag_idxs: Array
    tri_mask: Array
    lower: bool
    min_diag: float
    _arr: Array
    _log_diag: Array

    def __init__(self, loc: Array, arr: Array, lower: bool = True, min_diag: float = 1e-6):
        """
        Transformation of the form Ax + b, where A is a lower or upper triangular matrix. To
        ensure invertiblility, diagonal entries should be positive (and greater than min_diag).

        Args:
            loc (Array): Translation.
            arr (Array): Matrix.
            lower (bool, optional): Whether the mask should select the lower or upper triangular matrix (other elements ignored). Defaults to True.
            min_diag (float, optional): Minimum value on the diagonal, to ensure invertibility. Defaults to 1e-6.
        """
        
        if (arr.ndim != 2) or (arr.shape[0] != arr.shape[1]):
            ValueError("arr must be a square, 2-dimensional matrix.")
        if jnp.any(jnp.diag(arr) < min_diag):
            ValueError("arr diagonal entries must be greater than min_diag")
        
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
        
    @property
    def arr(self):
        "Get triangular array, (applies masking and min_diag constraint)."
        diag = jnp.exp(self._log_diag) + self.min_diag
        return (self.tri_mask*self._arr).at[self.diag_idxs].set(diag)

    def transform(self, x, condition = None):
        return self.arr @ x + self.loc

    def transform_and_log_abs_det_jacobian(self, x, condition = None):
        a = self.arr
        return a @ x + self.loc, jnp.log(jnp.diag(a)).sum()

    def inverse(self, y, condition = None):
        return solve_triangular(self.arr, y - self.loc, lower=self.lower)

    def inverse_and_log_abs_det_jacobian(self, y, condition = None):
        a = self.arr
        return solve_triangular(a, y - self.loc, lower=self.lower), -jnp.log(jnp.diag(a)).sum()


class AdditiveLinearCondition(Bijection):
    dim: int
    cond_dim: int
    W: Array

    def __init__(self, arr: Array):
        """Carries out `y = x + W @ condition`, as the forward transformation and 
        `x = y - W @ condition` as the inverse.

        Args:
            arr (Array): Array (W in the description.)
        """
        self.dim = arr.shape[0]
        self.cond_dim = arr.shape[1]
        self.W = arr
    
    def transform(self, x, condition = None):
        return x + self.W @ condition

    def transform_and_log_abs_det_jacobian(self, x, condition = None):
        return self.transform(x, condition), jnp.array(0)
        
    def inverse(self, y, condition = None):
        return y - self.W @ condition

    def inverse_and_log_abs_det_jacobian(self, y, condition = None):
        return self.inverse(y, condition), jnp.array(0)
