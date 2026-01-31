"""Rational quadratic spline bijections (https://arxiv.org/abs/1906.04032)."""

import jax
import jax.numpy as jnp
from jaxtyping import Array
from paramax import Parameterize, RealToIncreasingOnInterval
from paramax.utils import inv_softplus

from flowjax.bijections.bijection import AbstractBijection


class RationalQuadraticSpline(AbstractBijection):
    """Scalar RationalQuadraticSpline transformation (https://arxiv.org/abs/1906.04032).

    Args:
        knots: Number of inner knots.
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
    x_pos: Array
    y_pos: Array
    derivatives: Array
    shape = ()
    cond_shape = None

    def __init__(
        self,
        *,
        knots: int,
        interval: float | int | tuple[int | float, int | float],
        min_derivative: float = 1e-3,
        min_width: float | int = 1e-3,
    ):
        self.knots = knots
        interval = interval if isinstance(interval, tuple) else (-interval, interval)
        self.interval = interval
        self.x_pos = RealToIncreasingOnInterval(
            jnp.zeros(knots + 1),
            interval=interval,
            min_width=min_width,
            include_endpoints="both",
        )  # type: ignore
        self.y_pos = RealToIncreasingOnInterval(
            jnp.zeros(knots + 1),
            interval=interval,
            min_width=min_width,
            include_endpoints="both",
        )  # type: ignore
        self.derivatives = Parameterize(
            lambda arr: jax.nn.softplus(arr) + min_derivative,
            jnp.full(knots + 2, inv_softplus(1 - min_derivative)),
        )  # type: ignore

    def transform_and_log_det(self, x, condition=None):
        # Following notation from the paper
        x_pos, y_pos, derivatives = self.x_pos, self.y_pos, self.derivatives
        in_bounds = jnp.logical_and(
            x >= jnp.array(self.interval[0]),
            x <= jnp.array(self.interval[1]),
        )
        k = jnp.searchsorted(x_pos, x) - 1  # k is bin number
        k = jnp.clip(k, min=0, max=len(x_pos) - 2)
        xi = (x - x_pos[k]) / (x_pos[k + 1] - x_pos[k])
        sk = (y_pos[k + 1] - y_pos[k]) / (x_pos[k + 1] - x_pos[k])
        dk, dk1, yk, yk1 = derivatives[k], derivatives[k + 1], y_pos[k], y_pos[k + 1]
        num = (yk1 - yk) * (sk * xi**2 + dk * xi * (1 - xi))
        den = sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)
        y = yk + num / den  # eq. 4

        # avoid numerical precision issues transforming from in -> out of bounds
        y = jnp.clip(y, self.interval[0], self.interval[1])
        y = jnp.where(in_bounds, y, x)
        deriv = _rqs_derivative(
            dk=derivatives[k],
            dk1=derivatives[k + 1],
            sk=sk,
            xi=xi,
            in_bounds=in_bounds,
        )
        return y, jnp.log(deriv).sum()

    def inverse_and_log_det(self, y, condition=None):
        # Following notation from the paper
        x_pos, y_pos, derivatives = self.x_pos, self.y_pos, self.derivatives
        in_bounds = jnp.logical_and(
            y >= jnp.array(self.interval[0]),
            y <= jnp.array(self.interval[1]),
        )
        k = jnp.searchsorted(y_pos, y) - 1
        k = jnp.clip(k, min=0, max=len(x_pos) - 2)
        xk, xk1, yk, yk1 = x_pos[k], x_pos[k + 1], y_pos[k], y_pos[k + 1]
        sk = (yk1 - yk) / (xk1 - xk)
        y_delta_s_term = (y - yk) * (derivatives[k + 1] + derivatives[k] - 2 * sk)
        a = (yk1 - yk) * (sk - derivatives[k]) + y_delta_s_term
        b = (yk1 - yk) * derivatives[k] - y_delta_s_term
        c = -sk * (y - yk)
        sqrt_term = jnp.sqrt(b**2 - 4 * a * c)
        xi = (2 * c) / (-b - sqrt_term)
        x = xi * (xk1 - xk) + xk

        # avoid numerical precision issues transforming from in -> out of bounds
        x = jnp.clip(x, self.interval[0], self.interval[1])
        x = jnp.where(in_bounds, x, y)
        deriv = _rqs_derivative(
            dk=derivatives[k],
            dk1=derivatives[k + 1],
            sk=sk,
            xi=xi,
            in_bounds=in_bounds,
        )
        return x, -jnp.log(deriv).sum()


def _rqs_derivative(dk, dk1, sk, xi, in_bounds) -> Array:
    """The derivative dy/dx of the forward transformation."""
    # Following notation from the paper (eq. 5)
    num = sk**2 * (dk1 * xi**2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
    den = (sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)) ** 2
    derivative = jnp.where(in_bounds, num / den, 1.0)
    assert isinstance(derivative, Array)
    return derivative
