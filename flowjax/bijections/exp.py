"""Exponential bijection"""
import jax.numpy as jnp

from flowjax.bijections.bijection import Bijection


class Exp(Bijection):
    """Elementwise exponential transform (forward) and log transform (inverse)."""

    def __init__(self, shape: tuple[int, ...] = ()):
        """
        Args:
            shape (tuple[int, ...] | None): Shape of the bijection.
                Defaults to None.
        """
        self.shape = shape
        self.cond_shape = None

    def transform(self, x, condition=None):
        self._argcheck(x)
        return jnp.exp(x)

    def transform_and_log_det(self, x, condition=None):
        self._argcheck(x)
        return jnp.exp(x), x.sum()

    def inverse(self, y, condition=None):
        self._argcheck(y)
        return jnp.log(y)

    def inverse_and_log_det(self, y, condition=None):
        self._argcheck(y)
        x = jnp.log(y)
        return x, -x.sum()
