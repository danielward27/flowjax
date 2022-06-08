from flowjax.bijections.abc import ParameterisedBijection
import jax.numpy as jnp
import jax
from functools import partial


class RationalQuadraticSpline(ParameterisedBijection):

    def __init__(self, K, B, min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3):
        self.K = K
        self.B = B
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        pos_pad = jnp.zeros(self.K + 4)
        pad_idxs = jnp.array([0, 1, -2, -1])
        pad_vals = jnp.array(
            [-B * 1e4, -B, B, B * 1e4]
        )  # Avoids jax control flow for identity tails
        """
        RationalQuadraticSpline following Durkin et al. (2019),
        https://arxiv.org/abs/1906.04032. Each row of parameter matrices (x_pos,
        y_pos, derivatives) corresponds to a column (axis=1) in x. Ouside the interval
        [-B, B], the identity transform is used.

        Args:
            K (int): Number of inner knots B: (int):
            B: (int): Interval to transform [-B, B]
        """
        pos_pad = pos_pad.at[pad_idxs].set(pad_vals)
        self._pos_pad = pos_pad # End knots and beyond

    @property
    def pos_pad(self):
        return jax.lax.stop_gradient(self._pos_pad)

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def transform(self, x, x_pos, y_pos, derivatives):
        k = self._get_bin(x, x_pos)
        yk, yk1, xk, xk1 = y_pos[k], y_pos[k + 1], x_pos[k], x_pos[k + 1]
        dk, dk1 = derivatives[k], derivatives[k + 1]
        sk = (yk1 - yk) / (xk1 - xk)
        xi = (x - xk) / (xk1 - xk)
        return self._rational_quadratic(sk, xi, dk, dk1, yk, yk1)

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def inverse(self, y, x_pos, y_pos, derivatives):
        k = self._get_bin(y, y_pos)
        xk, xk1, yk = x_pos[k], x_pos[k + 1], y_pos[k]
        sk = (y_pos[k + 1] - yk) / (xk1 - xk)
        y_delta_s_term = (y - yk) * (derivatives[k + 1] + derivatives[k] - 2 * sk)
        a = (y_pos[k + 1] - yk) * (sk - derivatives[k]) + y_delta_s_term
        b = (y_pos[k + 1] - yk) * derivatives[k] - y_delta_s_term
        c = -sk * (y - yk)
        sqrt_term = jnp.sqrt(b ** 2 - 4 * a * c)
        xi = (2 * c) / (-b - sqrt_term)
        x = xi * (xk1 - xk) + xk
        return x

    def transform_and_log_abs_det_jacobian(self, x, x_pos, y_pos, derivatives):
        y, log_det = jax.vmap(self._transform_and_log_abs_det_jacobian)(
            x, x_pos, y_pos, derivatives
        )
        return y, log_det.sum()

    # Methods prepended with _ are defined for scalar x, and are vmapped as appropriate.
    def _transform_and_log_abs_det_jacobian(self, x, x_pos, y_pos, derivatives):
        "Defined for single dimensional x."
        k = self._get_bin(x, x_pos)
        yk, yk1, xk, xk1 = y_pos[k], y_pos[k + 1], x_pos[k], x_pos[k + 1]
        dk, dk1 = derivatives[k], derivatives[k + 1]
        sk = (yk1 - yk) / (xk1 - xk)
        xi = (x - xk) / (xk1 - xk)
        y = self._rational_quadratic(sk, xi, dk, dk1, yk, yk1)
        derivative = self._derivative(sk, xi, dk, dk1)
        return y, jnp.log(derivative)

    def num_params(self, dim: int):
        return (self.K * 3 - 1) * dim

    def get_args(self, params):
        params = params.reshape((-1, self.K * 3 - 1))
        return jax.vmap(self._get_args)(params)

    def _get_args(self, params):
        "Gets the arguments for a single dimension of x."
        widths = jax.nn.softmax(params[: self.K]) * 2 * self.B
        widths = self.min_bin_width + (1 - self.min_bin_width * self.K) * widths

        heights = jax.nn.softmax(params[self.K : self.K * 2]) * 2 * self.B
        heights = self.min_bin_height + (1 - self.min_bin_height * self.K) * heights

        derivatives = jax.nn.softplus(params[self.K * 2 :]) + self.min_derivative
        derivatives = jnp.pad(derivatives, 2, constant_values=1)
        x_pos = jnp.cumsum(widths) - self.B
        x_pos = self.pos_pad.at[2:-2].set(x_pos)
        y_pos = jnp.cumsum(heights) - self.B
        y_pos = self.pos_pad.at[2:-2].set(y_pos)
        return x_pos, y_pos, derivatives

    @staticmethod
    def _get_bin(target, positions):
        "Finds which bin/spline segment the target is in (defined for 1d)."
        cond1 = target <= positions[1:]
        cond2 = target > positions[:-1]
        return jnp.where(cond1 & cond2, size=1)[0][0]

    @staticmethod
    def _rational_quadratic(sk, xi, dk, dk1, yk, yk1):  # eq. 4
        num = (yk1 - yk) * (sk * xi ** 2 + dk * xi * (1 - xi))
        den = sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)
        return yk + num / den

    @staticmethod
    def _derivative(sk, xi, dk, dk1):  # eq. 5
        num = sk ** 2 * (dk1 * xi ** 2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
        den = (sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)) ** 2
        return num / den
