from paramax import AbstractUnwrappable, Parameterize
from flowjax.bijections.bijection import AbstractBijection
from jax import Array
import jax.numpy as jnp
import jax.nn as jnn
from jax.scipy import fft


class Householder(AbstractBijection):
    """A Householder reflection bijection.

    This bijection implements a Householder reflection, which is a linear
    transformation that reflects vectors across a hyperplane defined by a normal
    vector (params). The transformation is its own inverse and volume-preserving
    (determinant = ±1).

    Given a unit vector v, the transformation is:
    x → x - 2(x·v)v

    Attributes:
        shape: Shape of the input/output vectors
        cond_shape: Shape of conditional inputs (None as this bijection is unconditional)
        params: Normal vector defining the reflection hyperplane. The vector is
            normalized in the transformation, so scaling params will have no effect
            on the bijection.
    """
    shape: tuple[int, ...]
    unit_vec: Array | AbstractUnwrappable
    cond_shape = None

    def __init__(self, params: Array):
        self.shape = (params.shape[-1],)
        self.unit_vec = Parameterize(lambda x: x / jnp.linalg.norm(x), params)

    def _householder(self, x: Array) -> Array:
        return x - 2 * self.unit_vec * (x @ self.unit_vec)

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        return self._householder(x), jnp.zeros(())

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        return self._householder(y), jnp.zeros(())


class DiscreteCosine(AbstractBijection):
    """Discrete Cosine Transform (DCT) bijection.

    This bijection applies the DCT or its inverse along a specified axis.

    Attributes:
        shape: Shape of the input/output arrays
        cond_shape: Shape of conditional inputs (None as this bijection is unconditional)
        axis: Axis along which to apply the DCT
        norm: Normalization method, fixed to 'ortho' to ensure bijectivity
    """

    shape: tuple[int, ...]
    cond_shape = None
    axis: int

    def __init__(self, shape, *, axis: int = -1):
        self.shape = shape
        self.axis = axis

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        y = fft.dct(x, norm="ortho", axis=self.axis)
        return y, jnp.zeros(())

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        x = fft.idct(y, norm="ortho", axis=self.axis)
        return x, jnp.zeros(())
