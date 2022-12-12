from flowjax.bijections import Bijection
import jax.numpy as jnp
import jax


class Tanh(Bijection):
    """
    Tanh bijection.
    """

    cond_dim: int = 0

    def transform(self, x, condition=None):
        return jnp.tanh(x)

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        return jnp.tanh(x), self._log_abs_det_jac(x)

    def inverse(self, y, condition=None):
        return jnp.arctanh(y)

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        x = jnp.arctanh(y)
        return x, -self._log_abs_det_jac(x)

    def _log_abs_det_jac(self, x):
        return (-2 * (x + jax.nn.softplus(-2 * x) - jnp.log(2.0))).sum()
