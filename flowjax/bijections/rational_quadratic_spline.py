from flowjax.bijections.abc import ParameterisedBijection
import jax.numpy as jnp
import jax


class RationalQuadraticSpline1D(ParameterisedBijection):
    K: int
    B: float
    _pos_pad: jnp.ndarray  # End knots and beyond

    def __init__(self, K, B):
        self.K = K
        self.B = B
        pos_pad = jnp.zeros(self.K + 4)
        pad_idxs = jnp.array([0, 1, -2, -1])
        pad_vals = jnp.array(
            [-B * 1e4, -B, B, B * 1e4]
        )  # Avoids jax control flow for identity tails
        """One dimensional rational quadratic spline, following Durkin et al.
        (2019), https://arxiv.org/abs/1906.04032. RationalQuadraticSpline
        provides a vmapped version for multidimensional transformations. See
        RationalQuadraticSpline for more information.
        """

        pos_pad = pos_pad.at[pad_idxs].set(pad_vals)
        self._pos_pad = pos_pad

    @property  # TODO update to Buffer when introduced to eqx: https://github.com/patrick-kidger/equinox/issues/31
    def pos_pad(self):
        return jax.lax.stop_gradient(self._pos_pad)

    def transform(self, x, x_pos, y_pos, derivatives):
        k = self.get_bin(x, x_pos)
        yk, yk1, xk, xk1 = y_pos[k], y_pos[k + 1], x_pos[k], x_pos[k + 1]
        dk, dk1 = derivatives[k], derivatives[k + 1]
        sk = (yk1 - yk) / (xk1 - xk)
        xi = (x - xk) / (xk1 - xk)
        return self._rational_quadratic(sk, xi, dk, dk1, yk, yk1)

    def inverse(self, y, x_pos, y_pos, derivatives):
        k = self.get_bin(y, y_pos)
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
        k = self.get_bin(x, x_pos)
        yk, yk1, xk, xk1 = y_pos[k], y_pos[k + 1], x_pos[k], x_pos[k + 1]
        dk, dk1 = derivatives[k], derivatives[k + 1]
        sk = (yk1 - yk) / (xk1 - xk)
        xi = (x - xk) / (xk1 - xk)
        y = self._rational_quadratic(sk, xi, dk, dk1, yk, yk1)
        derivative = self._derivative(sk, xi, dk, dk1)
        return y, jnp.log(derivative)

    def num_params(self, dim: int = None):
        return self.K * 3 - 1

    def get_args(self, params):
        widths = jax.nn.softmax(params[: self.K]) * 2 * self.B
        heights = jax.nn.softmax(params[self.K : self.K * 2]) * 2 * self.B
        derivatives = jax.nn.softplus(params[self.K * 2 :])
        derivatives = jnp.pad(derivatives, 2, constant_values=1)
        x_pos = jnp.cumsum(widths) - self.B
        x_pos = self.pos_pad.at[2:-2].set(x_pos)
        y_pos = jnp.cumsum(heights) - self.B
        y_pos = self.pos_pad.at[2:-2].set(y_pos)
        return x_pos, y_pos, derivatives

    @staticmethod
    def get_bin(target, positions):
        cond1 = target <= positions[1:]
        cond2 = target > positions[:-1]
        return jnp.where(cond1 & cond2, size=1)[0][0]

    @staticmethod
    def _rational_quadratic(
        sk, xi, dk, dk1, yk, yk1
    ):  # eq. 4 https://arxiv.org/pdf/1906.04032.pdf
        num = (yk1 - yk) * (sk * xi ** 2 + dk * xi * (1 - xi))
        den = sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)
        return yk + num / den

    @staticmethod
    def _derivative(sk, xi, dk, dk1):  # eq. 5 https://arxiv.org/pdf/1906.04032.pdf
        num = sk ** 2 * (dk1 * xi ** 2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
        den = (sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)) ** 2
        return num / den


class RationalQuadraticSpline(ParameterisedBijection):
    K: int
    B: int
    spline: RationalQuadraticSpline1D

    def __init__(self, K, B):
        """
        RationalQuadraticSpline following Durkin et al. (2019),
        https://arxiv.org/abs/1906.04032. Each row of parameter matrices (x_pos,
        y_pos, derivatives) corresponds to a column (axis=1) in x

        Args:
            K (int): Number of inner knots B: (int): Interval to transform ([-B,
            B])
        """
        self.K = K
        self.B = B
        self.spline = RationalQuadraticSpline1D(K, B)

    def transform(self, x, x_pos, y_pos, derivatives):
        return jax.vmap(self.spline.transform)(x, x_pos, y_pos, derivatives)

    def transform_and_log_abs_det_jacobian(self, x, x_pos, y_pos, derivatives):
        y, log_abs_det_jacobian = jax.vmap(
            self.spline.transform_and_log_abs_det_jacobian
        )(x, x_pos, y_pos, derivatives)
        return y, log_abs_det_jacobian.sum()

    def inverse(self, y, x_pos, y_pos, derivatives):
        return jax.vmap(self.spline.inverse)(y, x_pos, y_pos, derivatives)

    def num_params(self, dim: int):
        return (self.K * 3 - 1) * dim

    def get_args(self, params):
        params = params.reshape((-1, self.K * 3 - 1))
        return jax.vmap(self.spline.get_args)(params)


# TODO maybe possible to make multilayer Spline+Permute with jax.scan?
