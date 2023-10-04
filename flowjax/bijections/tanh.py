"""Tanh bijection."""
import math
import warnings

import jax.numpy as jnp
from jax.nn import softplus

from flowjax.bijections.bijection import Bijection


def _tanh_log_grad(x):
    """log gradient vector of tanh transformation."""
    return -2 * (x + softplus(-2 * x) - jnp.log(2.0))


class Tanh(Bijection):
    """Tanh bijection."""

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        """
        Args:
            shape (tuple[int, ...] | None): Shape of the bijection. Defaults to None.
        """
        self.shape = shape
        self.cond_shape = None

    def transform(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return jnp.tanh(x)

    def transform_and_log_det(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return jnp.tanh(x), jnp.sum(_tanh_log_grad(x))

    def inverse(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        return jnp.arctanh(y)

    def inverse_and_log_det(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        x = jnp.arctanh(y)
        return x, -jnp.sum(_tanh_log_grad(x))


class LeakyTanh(Bijection):
    """
    Tanh bijection, with a linear transformation beyond +/- max_val. The value and
    gradient of the linear segments are set to match tanh at +/- max_val. This bijection
    can be useful to encourage values to be within an interval, whilst avoiding
    numerical precision issues, or in cases we require a real -> real mapping so Tanh
    is not appropriate.
    """

    max_val: float
    intercept: float
    linear_grad: float

    def __init__(self, max_val: float, shape: tuple[int, ...] = ()):
        """
        Args:
            max_val (float): Value above or below which the function becomes linear.
            shape (tuple[int, ...] | None): The shape of the bijection. Defaults to ().
        """
        self.max_val = float(max_val)
        self.linear_grad = math.exp(_tanh_log_grad(max_val))
        self.intercept = math.tanh(max_val) - self.linear_grad * max_val
        self.shape = shape
        self.cond_shape = None

    def transform(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        is_linear = jnp.abs(x) >= self.max_val
        linear_y = self.linear_grad * x + jnp.sign(x) * self.intercept
        tanh_y = jnp.tanh(x)
        return jnp.where(is_linear, linear_y, tanh_y)

    def transform_and_log_det(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        y = self.transform(x)
        log_grads = jnp.where(
            jnp.abs(x) >= self.max_val, jnp.log(self.linear_grad), _tanh_log_grad(x)
        )
        return y, jnp.sum(log_grads)

    def inverse(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        is_linear = jnp.abs(y) >= jnp.tanh(self.max_val)
        x_linear = (y - jnp.sign(y) * self.intercept) / self.linear_grad
        x_arctan = jnp.arctanh(y)
        return jnp.where(is_linear, x_linear, x_arctan)

    def inverse_and_log_det(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        x = self.inverse(y)
        log_grads = jnp.where(
            jnp.abs(y) >= jnp.tanh(self.max_val),
            jnp.log(self.linear_grad),
            _tanh_log_grad(x),
        )
        return x, -jnp.sum(log_grads)


def TanhLinearTails(*args, **kwargs):
    warnings.warn(
        "This class has been renamed to LeakyTanh and TanhLinearTails will be removed. "
        "please update to the new name.",
        stacklevel=2,
    )
    return LeakyTanh(*args, **kwargs)
