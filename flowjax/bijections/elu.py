"""Exponential linear unit (ELU) bijection"""
import jax.numpy as jnp

from flowjax.bijections.bijection import Bijection
from flowjax.bijections.chain import Chain
from flowjax.bijections.affine import Affine


class ELU(Bijection):
    """Exponential linear unit (ELU) bijection: exp(x)-1 for x<0 and x for x>=0."""

    def __init__(self, shape: tuple[int, ...] = ()):
        """
        Args:
            shape (tuple[int, ...] | None): Shape of the bijection.
                Defaults to None.
        """
        self.shape = shape
        self.cond_shape = None

    def transform(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return jnp.where(x < 0, jnp.exp(x)-1, x)

    def transform_and_log_det(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return self.transform(x), jnp.where(x < 0, x, 0).sum()

    def inverse(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        return jnp.where(y < 0, jnp.log(y+1), y)

    def inverse_and_log_det(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        x = self.inverse(y)
        return x, jnp.where(y < 0, -x, 0).sum()


def OnePlusELU(shape: tuple[int, ...] = ()):
    """Strictly positive exponential linear unit (ELU) bijection: exp(x) for x<0 and x+1 for x>=0.
    
    Args:
    shape (tuple[int, ...] | None): Shape of the bijection.
        Defaults to None.
    """
    bijections = [ELU(shape), Affine(loc=jnp.ones(shape))]
    return Chain(bijections)
