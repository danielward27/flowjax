
from flowjax.bijections import Bijection
import jax.numpy as jnp

class Exp(Bijection):
    def __init__(self):
        """Elementwise exponential transform (forward) and log transform (inverse)."""
        self.shape = None
        self.cond_shape = None

    def transform(self, x, condition = None):
        return jnp.exp(x)

    def transform_and_log_abs_det_jacobian(self, x, condition = None):
        return jnp.exp(x), x.sum()

    def inverse(self, y, condition = None):
        return jnp.log(y)

    def inverse_and_log_abs_det_jacobian(self, y, condition = None):
        x = jnp.log(y)
        return x, -x.sum()
