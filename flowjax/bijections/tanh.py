"""Tanh bijection."""

import math
from typing import ClassVar

import jax.numpy as jnp
from jax.nn import softplus

from flowjax.bijections.bijection import AbstractBijection


def _tanh_log_grad(x):
    # Log gradient vector of tanh transformation.
    return -2 * (x + softplus(-2 * x) - jnp.log(2.0))


class Tanh(AbstractBijection):
    r"""Tanh bijection :math:`y=\tanh(x)`."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def transform(self, x, condition=None):
        return jnp.tanh(x)

    def transform_and_log_det(self, x, condition=None):
        return jnp.tanh(x), jnp.sum(_tanh_log_grad(x))

    def inverse(self, y, condition=None):
        return jnp.arctanh(y)

    def inverse_and_log_det(self, y, condition=None):
        x = jnp.arctanh(y)
        return x, -jnp.sum(_tanh_log_grad(x))


class LeakyTanh(AbstractBijection):
    """Tanh bijection, with a linear transformation beyond +/- max_val.

    The value and gradient of the linear segments are set to match tanh at +/- max_val.
    This bijection can be useful to encourage values to be within an interval, whilst
    avoiding numerical precision issues, or in cases we require a real -> real mapping
    so Tanh is not appropriate.

    Args:
        max_val: Value above or below which the function becomes linear.
        shape: The shape of the bijection. Defaults to ().
    """

    max_val: float
    intercept: float
    linear_grad: float
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def __init__(self, max_val: float | int, shape: tuple[int, ...] = ()):
        self.max_val = float(max_val)
        self.linear_grad = math.exp(_tanh_log_grad(max_val))
        self.intercept = math.tanh(max_val) - self.linear_grad * max_val
        self.shape = shape

    def transform(self, x, condition=None):
        is_linear = jnp.abs(x) >= self.max_val
        linear_y = self.linear_grad * x + jnp.sign(x) * self.intercept
        tanh_y = jnp.tanh(x)
        return jnp.where(is_linear, linear_y, tanh_y)

    def transform_and_log_det(self, x, condition=None):
        y = self.transform(x)
        log_grads = jnp.where(
            jnp.abs(x) >= self.max_val,
            jnp.log(self.linear_grad),
            _tanh_log_grad(x),
        )
        return y, jnp.sum(log_grads)

    def inverse(self, y, condition=None):
        is_linear = jnp.abs(y) >= jnp.tanh(self.max_val)
        x_linear = (y - jnp.sign(y) * self.intercept) / self.linear_grad
        x_arctan = jnp.arctanh(y)
        return jnp.where(is_linear, x_linear, x_arctan)

    def inverse_and_log_det(self, y, condition=None):
        x = self.inverse(y)
        log_grads = jnp.where(
            jnp.abs(y) >= jnp.tanh(self.max_val),
            jnp.log(self.linear_grad),
            _tanh_log_grad(x),
        )
        return x, -jnp.sum(log_grads)
