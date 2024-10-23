"""Rational quadratic spline bijections (https://arxiv.org/abs/1906.04032)."""

from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from paramax import AbstractUnwrappable, Parameterize
from paramax.utils import inv_softplus

from flowjax.bijections.bijection import AbstractBijection


def _real_to_increasing_on_interval(
    arr: Float[Array, " dim"],
    interval: tuple[int | float, int | float],
    softmax_adjust: float = 1e-2,
    *,
    pad_with_ends: bool = True,
):
    """Transform unconstrained vector to monotonically increasing positions on [-B, B].

    Args:
        arr: Parameter vector.
        interval: Interval to transform output. Defaults to 1.
        softmax_adjust : Rescales softmax output using
            ``(widths + softmax_adjust/widths.size) / (1 + softmax_adjust)``. e.g.
            0=no adjustment, 1=average softmax output with evenly spaced widths, >1
            promotes more evenly spaced widths.
        pad_with_ends: Whether to pad the with -interval and interval. Defaults to True.
    """
    if softmax_adjust < 0:
        raise ValueError("softmax_adjust should be >= 0.")

    widths = jax.nn.softmax(arr)
    widths = (widths + softmax_adjust / widths.size) / (1 + softmax_adjust)
    widths = widths.at[0].set(widths[0] / 2)
    scale = interval[1] - interval[0]
    pos = interval[0] + scale * jnp.cumsum(widths)

    if pad_with_ends:
        pos = jnp.pad(pos, pad_width=1, constant_values=interval)

    return pos


class RationalQuadraticSpline(AbstractBijection):
    """Scalar RationalQuadraticSpline transformation (https://arxiv.org/abs/1906.04032).

    Args:
        knots: Number of knots.
        interval: Interval to transform, if a scalar value, uses [-interval, interval],
            if a tuple, uses [interval[0], interval[1]]
        min_derivative: Minimum dervivative. Defaults to 1e-3.
        softmax_adjust: Controls minimum bin width and height by rescaling softmax
            output, e.g. 0=no adjustment, 1=average softmax output with evenly spaced
            widths, >1 promotes more evenly spaced widths. See
            ``real_to_increasing_on_interval``. Defaults to 1e-2.
    """

    knots: int
    interval: tuple[int | float, int | float]
    softmax_adjust: float | int
    min_derivative: float
    x_pos: Array | AbstractUnwrappable[Array]
    y_pos: Array | AbstractUnwrappable[Array]
    derivatives: Array | AbstractUnwrappable[Array]
    shape: ClassVar[tuple] = ()
    cond_shape: ClassVar[None] = None

    def __init__(
        self,
        *,
        knots: int,
        interval: float | int | tuple[int | float, int | float],
        min_derivative: float = 1e-3,
        softmax_adjust: float | int = 1e-2,
    ):
        self.knots = knots
        interval = interval if isinstance(interval, tuple) else (-interval, interval)
        self.interval = interval
        self.softmax_adjust = softmax_adjust
        self.min_derivative = min_derivative

        to_interval = jnp.vectorize(
            partial(
                _real_to_increasing_on_interval,
                interval=interval,
                softmax_adjust=softmax_adjust,
            ),
            signature="(a)->(b)",
        )

        self.x_pos = Parameterize(to_interval, jnp.zeros(knots))
        self.y_pos = Parameterize(to_interval, jnp.zeros(knots))
        self.derivatives = Parameterize(
            lambda arr: jax.nn.softplus(arr) + min_derivative,
            jnp.full(knots + 2, inv_softplus(1 - min_derivative)),
        )

    def transform_and_log_det(self, x, condition=None):
        # Following notation from the paper
        x_pos, y_pos, derivatives = self.x_pos, self.y_pos, self.derivatives
        in_bounds = jnp.logical_and(x >= self.interval[0], x <= self.interval[1])
        x_robust = jnp.where(in_bounds, x, 0)  # To avoid nans
        k = jnp.searchsorted(x_pos, x_robust) - 1  # k is bin number
        xi = (x_robust - x_pos[k]) / (x_pos[k + 1] - x_pos[k])
        sk = (y_pos[k + 1] - y_pos[k]) / (x_pos[k + 1] - x_pos[k])
        dk, dk1, yk, yk1 = derivatives[k], derivatives[k + 1], y_pos[k], y_pos[k + 1]
        num = (yk1 - yk) * (sk * xi**2 + dk * xi * (1 - xi))
        den = sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)
        y = yk + num / den  # eq. 4

        # avoid numerical precision issues transforming from in -> out of bounds
        y = jnp.clip(y, self.interval[0], self.interval[1])
        y = jnp.where(in_bounds, y, x)

        return y, jnp.log(self.derivative(x)).sum()

    def inverse_and_log_det(self, y, condition=None):
        # Following notation from the paper
        x_pos, y_pos, derivatives = self.x_pos, self.y_pos, self.derivatives
        in_bounds = jnp.logical_and(y >= self.interval[0], y <= self.interval[1])
        y_robust = jnp.where(in_bounds, y, 0)  # To avoid nans
        k = jnp.searchsorted(y_pos, y_robust) - 1
        xk, xk1, yk, yk1 = x_pos[k], x_pos[k + 1], y_pos[k], y_pos[k + 1]
        sk = (yk1 - yk) / (xk1 - xk)
        y_delta_s_term = (y_robust - yk) * (
            derivatives[k + 1] + derivatives[k] - 2 * sk
        )
        a = (yk1 - yk) * (sk - derivatives[k]) + y_delta_s_term
        b = (yk1 - yk) * derivatives[k] - y_delta_s_term
        c = -sk * (y_robust - yk)
        sqrt_term = jnp.sqrt(b**2 - 4 * a * c)
        xi = (2 * c) / (-b - sqrt_term)
        x = xi * (xk1 - xk) + xk

        # avoid numerical precision issues transforming from in -> out of bounds
        x = jnp.clip(x, self.interval[0], self.interval[1])
        x = jnp.where(in_bounds, x, y)

        return x, -jnp.log(self.derivative(x)).sum()

    def derivative(self, x) -> Array:
        """The derivative dy/dx of the forward transformation."""
        # Following notation from the paper (eq. 5)
        x_pos, y_pos, derivatives = self.x_pos, self.y_pos, self.derivatives
        in_bounds = jnp.logical_and(x >= self.interval[0], x <= self.interval[1])
        x_robust = jnp.where(in_bounds, x, 0)  # To avoid nans
        k = jnp.searchsorted(x_pos, x_robust) - 1
        xi = (x_robust - x_pos[k]) / (x_pos[k + 1] - x_pos[k])
        sk = (y_pos[k + 1] - y_pos[k]) / (x_pos[k + 1] - x_pos[k])
        dk, dk1 = derivatives[k], derivatives[k + 1]
        num = sk**2 * (dk1 * xi**2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
        den = (sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)) ** 2
        derivative = num / den
        return jnp.where(in_bounds, derivative, 1.0)
