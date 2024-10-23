"""Power transform."""

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp

from flowjax.bijections.bijection import AbstractBijection


class Power(AbstractBijection):
    """Power transform :math:`y = x^p`.

    Supports postive values, over which this is a bijection.

    Args:
        exponent: The exponent.
        shape: The shape of the bijection.
    """

    exponent: int | float
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def transform_and_log_det(self, x, condition=None):
        x = eqx.error_if(x, x < 0, "Negative values not supported for Power.")
        y = x**self.exponent
        return y, jnp.log(jnp.abs(self.exponent * y / x)).sum()

    def inverse_and_log_det(self, y, condition=None):
        y = eqx.error_if(y, y < 0, "Negative values not supported for Power.")
        x = y ** (1 / self.exponent)
        return x, -jnp.log(jnp.abs(self.exponent * y / x)).sum()
