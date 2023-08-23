"""SoftPlus bijection"""
import jax.numpy as jnp
from jax.nn import softplus

from flowjax.bijections.bijection import Bijection


class SoftPlus(Bijection):
    r"""Transforms to positive domain using softplus :math:`y = \log(1 + \exp(x))`."""

    def __init__(self, shape: tuple[int] = ()):
        self.shape = shape
        self.cond_shape = None

    def transform(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return softplus(x)

    def transform_and_log_det(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return softplus(x), -softplus(-x).sum()

    def inverse(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        return jnp.log(-jnp.expm1(-y)) + y

    def inverse_and_log_det(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        x = self.inverse(y)
        return x, softplus(-x).sum()
