import equinox as eqx
import jax.numpy as jnp
from jax import random
import jax
from realnvp.bijections.coupling import Coupling
from realnvp.bijections.affine import Affine
from realnvp.bijections.permute import Permute


class RealNVP(eqx.Module):
    layers: list

    def __init__(
        self,
        key: random.PRNGKey,
        D: int,
        conditioner_width: int,
        conditioner_depth: int,
        num_layers: int,  # add option for other bijections?
    ):

        layers = []
        ds = [round(jnp.floor(D / 2).item()), round(jnp.ceil(D / 2).item())]
        permutation = jnp.flip(jnp.arange(D))
        for i in range(num_layers):
            key, subkey = random.split(key)
            d = ds[0] if i % 2 == 0 else ds[1]
            layers.extend(
                [
                    Coupling(key, Affine(), d, D, conditioner_width, conditioner_depth),
                    Permute(permutation),
                ]
            )
        self.layers = layers[:-2]  # remove last leakyRelu and permute

    def transform(self, x):
        z = x
        for layer in self.layers:
            z = layer.transform(z)
        return z

    def transform_and_log_abs_det_jacobian(self, x):
        log_abs_det_jac = 0
        z = x
        for layer in self.layers:
            z, log_abs_det_jac_i = layer.transform_and_log_abs_det_jacobian(z)
            log_abs_det_jac += log_abs_det_jac_i
        return z, log_abs_det_jac

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x):
        """Log probability in target distribution, assuming a standard normal
        base distribution."""
        z, log_abs_det = self.transform_and_log_abs_det_jacobian(x)
        p_z = jax.scipy.stats.norm.logpdf(z)
        return (p_z + log_abs_det).mean()

