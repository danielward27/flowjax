from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
from flowjax.utils import Array, real_to_increasing_on_interval
from flowjax.bijections import Bijection
from flowjax.bijections.jax_transforms import Vmap
import equinox as eqx


class _ScalarRationalQuadraticSpline(Bijection):
    """Scaler RationalQuadraticSpline transformation (https://arxiv.org/abs/1906.04032)."""

    knots: int
    interval: float
    softmax_adjust: float
    min_derivative: float
    unbounded_x_pos: Array
    unbounded_y_pos: Array
    unbounded_derivatives: Array

    def __init__(
        self,
        knots: int,
        interval: float,
        min_derivative: float = 1e-3,
        softmax_adjust: float = 1e-2,
    ):
        self.knots = knots
        self.interval = interval
        self.softmax_adjust = softmax_adjust
        self.min_derivative = min_derivative
        self.shape = ()
        self.cond_shape = None

        # Inexact arrays
        self.unbounded_x_pos = jnp.zeros(knots)
        self.unbounded_y_pos = jnp.zeros(knots)
        self.unbounded_derivatives = jnp.full(
            knots + 2, jnp.log(jnp.exp(1 - min_derivative) - 1)
        )

    @property
    def x_pos(self):
        "Get the knot x positions."
        x_pos = real_to_increasing_on_interval(
            self.unbounded_x_pos, self.interval, self.softmax_adjust
        )
        return jnp.pad(x_pos, 1, constant_values=(-self.interval, self.interval))

    @property
    def y_pos(self):
        "Get the knot y positions."
        y_pos = real_to_increasing_on_interval(
            self.unbounded_y_pos, self.interval, self.softmax_adjust
        )
        return jnp.pad(y_pos, 1, constant_values=(-self.interval, self.interval))

    @property
    def derivatives(self):
        "Get the knot derivitives"
        return jax.nn.softplus(self.unbounded_derivatives) + self.min_derivative

    def transform(self, x, condition=None):
        x_pos, y_pos, derivatives = self.x_pos, self.y_pos, self.derivatives
        in_bounds = jnp.logical_and(x > -self.interval, x < self.interval)
        x_robust = jnp.where(in_bounds, x, 0)  # To avoid nans
        k = jnp.searchsorted(x_pos, x_robust) - 1  # k is bin number
        xi = (x_robust - x_pos[k]) / (x_pos[k + 1] - x_pos[k])
        sk = (y_pos[k + 1] - y_pos[k]) / (x_pos[k + 1] - x_pos[k])
        dk, dk1, yk, yk1 = derivatives[k], derivatives[k + 1], y_pos[k], y_pos[k + 1]
        num = (yk1 - yk) * (sk * xi**2 + dk * xi * (1 - xi))
        den = sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)
        y = yk + num / den  # eq. 4
        return jnp.where(in_bounds, y, x)

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        y = self.transform(x)
        derivative = self.derivative(x)
        return y, jnp.log(derivative).sum()

    def inverse(self, y, condition=None):
        x_pos, y_pos, derivatives = self.x_pos, self.y_pos, self.derivatives
        in_bounds = jnp.logical_and(y > -self.interval, y < self.interval)

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
        return jnp.where(in_bounds, x, y)

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        x = self.inverse(y)
        derivative = self.derivative(x)
        return x, -jnp.log(derivative).sum()

    def derivative(self, x):  # eq. 5
        x_pos, y_pos, derivatives = self.x_pos, self.y_pos, self.derivatives
        in_bounds = jnp.logical_and(x > -self.interval, x < self.interval)
        x_robust = jnp.where(in_bounds, x, 0)  # To avoid nans
        k = jnp.searchsorted(x_pos, x_robust) - 1
        xi = (x_robust - x_pos[k]) / (x_pos[k + 1] - x_pos[k])
        sk = (y_pos[k + 1] - y_pos[k]) / (x_pos[k + 1] - x_pos[k])
        dk, dk1 = derivatives[k], derivatives[k + 1]
        num = sk**2 * (dk1 * xi**2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
        den = (sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)) ** 2
        derivative = num / den
        return jnp.where(in_bounds, derivative, 1.0)


class RationalQuadraticSpline(Vmap):
    def __init__(
        self,
        knots: int,
        interval: float,
        shape: Tuple[int] = (),
        min_derivative: float = 1e-3,
        softmax_adjust: float = 1e-2,
    ) -> None:
        """Elementwise rational quadratic spline transform (https://arxiv.org/abs/1906.04032),
        initialised at the identity function.

        Args:
            knots (int): Number of knots.
            interval (float): interval to transform, [-interval, interval].
            shape (Tuple[int], optional): Shape of transformation. Defaults to ().
            min_derivative (float, optional): Minimum dervivative. Defaults to 1e-3.
            softmax_adjust (float, optional): Controls minimum bin width and height by
                rescaling softmax output, e.g. 0=no adjustment, 1=average softmax output
                with evenly spaced widths, >1 promotes more evenly spaced widths.
                See ``real_to_increasing_on_interval``.. Defaults to 1e-2.
        """

        def constructor(dummy):  # Dummy variable to vmap over
            return _ScalarRationalQuadraticSpline(
                knots, interval, min_derivative, softmax_adjust
            )

        # Create constructor with appropriate number of batch dimensions
        for _ in shape:
            constructor = eqx.filter_vmap(constructor)

        spline = constructor(jnp.zeros(shape))
        super().__init__(spline, shape)
