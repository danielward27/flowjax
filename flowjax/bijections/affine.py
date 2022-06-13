from flowjax.bijections.abc import ParameterisedBijection
import jax.numpy as jnp

class Affine(ParameterisedBijection):
    "Affine transformation compatible with neural network parameterisation."
    def transform(self, x, loc, log_scale):
        return x * jnp.exp(log_scale) + loc

    def transform_and_log_abs_det_jacobian(self, x, loc, log_scale):
        return x * jnp.exp(log_scale) + loc, jnp.sum(log_scale)

    def inverse(self, y, loc, log_scale):
        return (y - loc) / jnp.exp(log_scale)

    def num_params(self, dim: int):
        return dim * 2

    def get_args(self, params):
        loc, log_scale = params.split(2)
        return loc, log_scale
