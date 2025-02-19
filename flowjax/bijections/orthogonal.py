"""Orthogonal transformations."""

import jax.numpy as jnp
from jax import Array
from jax.scipy import fft
from jaxtyping import ArrayLike
from paramax import AbstractUnwrappable, Parameterize

from flowjax.bijections.bijection import AbstractBijection
from flowjax.utils import arraylike_to_array


class Householder(AbstractBijection):
    """A Householder reflection.

    A linear transformation reflecting vectors across a hyperplane defined by a normal
    vector (params). The transformation is its own inverse and volume-preserving
    (determinant = -1). Given a unit vector :math:`v`, the transformation is
    :math:`y = x - 2(x^T v)v`.

    It is often desirable to stack multiple such transforms (e.g. up to the
    dimensionality of the data):

    .. doctest::

        >>> from flowjax.bijections import Householder, Scan
        >>> import jax.random as jr
        >>> import equinox as eqx
        >>> import jax.numpy as jnp

        >>> dim = 5
        >>> keys = jr.split(jr.key(0), dim)
        >>> householder_stack = Scan(
        ...    eqx.filter_vmap(lambda key: Householder(jr.normal(key, dim)))(keys)
        ... )

    Args:
        params: Normal vector defining the reflection hyperplane. The vector is
            normalized in the transformation, so scaling params will have no effect
            on the bijection.
    """

    shape: tuple[int, ...]
    unit_vec: Array | AbstractUnwrappable
    cond_shape = None

    def __init__(self, params: ArrayLike):
        params = arraylike_to_array(params)
        if params.ndim != 1:
            raise ValueError("params must be a vector.")
        self.shape = params.shape
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

    Args:
        shape: Shape of the input/output arrays.
        axis: Axis along which to apply the DCT.
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
