"""Contains transformers, which are bijections that have methods that facilitate
parameterisation with neural networks. All transformers have the "Transformer"
suffix, to avoid potential name clashes with bijections.
"""

from functools import partial

import jax
import jax.numpy as jnp

from flowjax.utils import Array
from typing import List
from abc import ABC, abstractmethod

from equinox import Module

from flowjax.utils import Array



class Transformer(ABC, Module):
    """Bijection which facilitates parameterisation with a neural network output
    (e.g. as in coupling flows, or masked autoressive flows). Should not contain
    (directly) trainable parameters."""

    @abstractmethod
    def transform(self, x: Array, *args: Array) -> Array:
        """Apply transformation."""

    @abstractmethod
    def transform_and_log_abs_det_jacobian(self, x: Array, *args: Array) -> tuple:
        """Apply transformation and compute log absolute value of the Jacobian determinant."""

    @abstractmethod
    def inverse(self, y: Array, *args: Array) -> Array:
        """Invert the transformation."""

    def inverse_and_log_abs_det_jacobian(self, x: Array, *args: Array) -> tuple:
        """Invert the transformation and compute the log absolute value of the Jacobian determinant."""

    @abstractmethod
    def num_params(self, dim: int) -> int:
        "Total number of parameters required for bijection."

    @abstractmethod
    def get_ranks(self, dim: int) -> Array:
        "The ranks of the parameters, i.e. which dimension of the input the parameters correspond to."

    @abstractmethod
    def get_args(self, params: Array) -> List[Array]:
        "Transform unconstrained vector of params (e.g. nn output) into args for transformation."


class AffineTransformer(Transformer):
    "Affine transformation compatible with neural network parameterisation."

    def transform(self, x, loc, scale):
        return x * scale + loc

    def transform_and_log_abs_det_jacobian(self, x, loc, scale):
        return x * scale + loc, jnp.log(scale).sum()

    def inverse(self, y, loc, scale):
        return (y - loc) / scale

    def inverse_and_log_abs_det_jacobian(self, y, loc, scale):
        return self.inverse(y, loc, scale), -jnp.log(scale).sum()

    def num_params(self, dim):
        return dim * 2

    def get_ranks(self, dim):
        return jnp.tile(jnp.arange(dim), 2)

    def get_args(self, params):
        loc, log_scale = params.split(2)
        return loc, jnp.exp(log_scale)


class RationalQuadraticSplineTransformer(Transformer):
    """RationalQuadraticSplineTransformer (https://arxiv.org/abs/1906.04032)."""
    K: int
    B: int
    softmax_adjust: float
    min_derivative: float

    def __init__(self, K, B, softmax_adjust=1e-2, min_derivative=1e-3):
        """
        Each row of parameter matrices (x_pos, y_pos, derivatives) corresponds to a column in x.
        Ouside the interval [-B, B], the identity transform is used. 

        Args:
            K (int): Number of inner knots
            B: (int): Interval to transform [-B, B]
            softmax_adjust: (float): Controls minimum bin width and height by rescaling softmax output, e.g. 0=no adjustment, 1=average softmax output with evenly spaced widths, >1 promotes more evenly spaced widths. See `real_to_increasing_on_interval`.
            min_derivative: (float): Minimum derivative.
        """
        self.K = K
        self.B = B
        self.softmax_adjust = softmax_adjust
        self.min_derivative = min_derivative

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def transform(self, x, x_pos, y_pos, derivatives):
        k = jnp.searchsorted(x_pos, x) - 1
        xi = (x - x_pos[k]) / (x_pos[k + 1] - x_pos[k])
        sk = (y_pos[k + 1] - y_pos[k]) / (x_pos[k + 1] - x_pos[k])
        dk, dk1, yk, yk1 = derivatives[k], derivatives[k + 1], y_pos[k], y_pos[k + 1]
        num = (yk1 - yk) * (sk * xi**2 + dk * xi * (1 - xi))
        den = sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)
        return yk + num / den  # eq. 4

    def transform_and_log_abs_det_jacobian(self, x, x_pos, y_pos, derivatives):
        y = self.transform(x, x_pos, y_pos, derivatives)
        derivative = self.derivative(x, x_pos, y_pos, derivatives)
        return y, jnp.log(derivative).sum()

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def inverse(self, y, x_pos, y_pos, derivatives):
        k = jnp.searchsorted(y_pos, y) - 1
        xk, xk1, yk, yk1 = x_pos[k], x_pos[k + 1], y_pos[k], y_pos[k + 1]
        sk = (yk1 - yk) / (xk1 - xk)
        y_delta_s_term = (y - yk) * (derivatives[k + 1] + derivatives[k] - 2 * sk)
        a = (yk1 - yk) * (sk - derivatives[k]) + y_delta_s_term
        b = (yk1 - yk) * derivatives[k] - y_delta_s_term
        c = -sk * (y - yk)
        sqrt_term = jnp.sqrt(b**2 - 4 * a * c)
        xi = (2 * c) / (-b - sqrt_term)
        x = xi * (xk1 - xk) + xk
        return x

    def inverse_and_log_abs_det_jacobian(self, y, x_pos, y_pos, derivatives):
        x = self.inverse(y, x_pos, y_pos, derivatives)
        derivative = self.derivative(x, x_pos, y_pos, derivatives)
        return x, -jnp.log(derivative).sum()

    def num_params(self, dim: int):
        return (self.K * 3 - 1) * dim

    def get_ranks(self, dim: int):
        return jnp.repeat(jnp.arange(dim), self.K * 3 - 1)

    def get_args(self, params):
        params = params.reshape((-1, self.K * 3 - 1))
        return jax.vmap(self._get_args)(params)

    def _get_args(self, params):
        "Gets the arguments for a single dimension of x (defined for 1d)."
        x_pos = real_to_increasing_on_interval(
            params[: self.K], self.B, self.softmax_adjust
        )
        y_pos = real_to_increasing_on_interval(
            params[self.K : self.K * 2], self.B, self.softmax_adjust
        )
        derivatives = jax.nn.softplus(params[self.K * 2 :]) + self.min_derivative

        # Padding sets up linear spline from the edge of the bounding box to B * 1e4
        pos_pad = jnp.array([self.B, 1e4 * self.B])
        x_pos = jnp.hstack((-jnp.flip(pos_pad), x_pos, pos_pad))
        y_pos = jnp.hstack((-jnp.flip(pos_pad), y_pos, pos_pad))
        derivatives = jnp.pad(derivatives, 2, constant_values=1)
        return x_pos, y_pos, derivatives

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def derivative(self, x, x_pos, y_pos, derivatives):  # eq. 5
        k = jnp.searchsorted(x_pos, x) - 1
        xi = (x - x_pos[k]) / (x_pos[k + 1] - x_pos[k])
        sk = (y_pos[k + 1] - y_pos[k]) / (x_pos[k + 1] - x_pos[k])
        dk, dk1 = derivatives[k], derivatives[k + 1]
        num = sk**2 * (dk1 * xi**2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
        den = (sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)) ** 2
        return num / den


def real_to_increasing_on_interval(
    arr: Array, B: float = 1, softmax_adjust: float = 1e-2
):
    """Transform unconstrained parameter vector to monotonically increasing positions on [-B, B].

    Args:
        arr (Array): Parameter vector.
        B (float, optional): Interval to transform output. Defaults to 1.
        softmax_adjust (float, optional): Rescales softmax output using (widths + softmax_adjust/widths.size) / (1 + softmax_adjust). e.g. 0=no adjustment, 1=average softmax output with evenly spaced widths, >1 promotes more evenly spaced widths.
    """
    if softmax_adjust < 0:
        raise ValueError("softmax_adjust should be >= 0.")
    widths = jax.nn.softmax(arr)
    widths = (widths + softmax_adjust / widths.size) / (1 + softmax_adjust)
    widths = widths.at[0].set(widths[0] / 2)
    return 2 * B * jnp.cumsum(widths) - B
