from flowjax.bijections.abc import Bijection
from flowjax.utils import Array
import jax.numpy as jnp


class Affine(Bijection):
    loc: Array
    log_scale: Array

    def __init__(self, loc: Array, scale: Array):
        "Elementwise affine transformation. Condition is ignored."
        self.loc = loc
        self.log_scale = jnp.log(scale)
        self.cond_dim = 0

    def transform(self, x, condition = None):
        return x * self.scale + self.loc

    def transform_and_log_abs_det_jacobian(self, x, condition = None):
        return x * self.scale + self.loc, self.log_scale.sum()

    def inverse(self, y, condition = None):
        return (y - self.loc) / self.scale

    def inverse_and_log_abs_det_jacobian(self, y, condition = None):
        return (y - self.loc) / self.scale, -self.log_scale.sum()

    @property
    def scale(self):
        return jnp.exp(self.log_scale)

        