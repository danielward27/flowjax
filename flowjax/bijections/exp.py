"""Exponential bijection."""

from typing import ClassVar

import jax.numpy as jnp

from flowjax.bijections.bijection import AbstractBijection


class Exp(AbstractBijection):
    """Elementwise exponential transform (forward) and log transform (inverse).

    Args:
        shape: Shape of the bijection. Defaults to ().
    """

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def transform_and_log_det(self, x, condition=None):
        return jnp.exp(x), x.sum()

    def inverse_and_log_det(self, y, condition=None):
        x = jnp.log(y)
        return x, -x.sum()
