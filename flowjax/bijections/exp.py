"""Exponential bijection"""
from typing import Optional
import jax.numpy as jnp

from flowjax.bijections.bijection import Bijection


class Exp(Bijection):
    """Elementwise exponential transform (forward) and log transform (inverse)."""

    def __init__(self, shape: Optional[tuple[int, ...]] = None):
        """
        Args:
            shape (Optional[tuple[int, ...]], optional): Shape of the bijection.
                Defaults to None.
        """
        self.shape = shape
        self.cond_shape = None

    def transform(self, x, condition=None):
        self._argcheck(x)
        return jnp.exp(x)

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        self._argcheck(x)
        return jnp.exp(x), x.sum()

    def inverse(self, y, condition=None):
        self._argcheck(y)
        return jnp.log(y)

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        self._argcheck(y)
        x = jnp.log(y)
        return x, -x.sum()
