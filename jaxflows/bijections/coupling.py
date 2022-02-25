from jaxflows.bijections.abc import ParameterisedBijection
import equinox as eqx
import jax.numpy as jnp
from jax import random
from jaxflows.bijections.permute import Permute
from jaxflows.bijections.abc import Bijection

class Coupling(Bijection, eqx.Module):
    d: int
    D: int
    bijection: ParameterisedBijection
    conditioner: eqx.nn.MLP

    def __init__(
        self,
        key: random.PRNGKey,
        bijection : ParameterisedBijection,
        d: int,
        D: int,
        conditioner_width: int,
        conditioner_depth: int,
    ):
        """Coupling layer implementation.

        Args:
            key (random.PRNGKey): Key
            bijection (ParameterisedBijection): Bijection to be parameterised by the conditioner neural netork.
            d (int): Number of untransformed conditioning variables.
            D (int): Total dimension.
            conditioner_width (int): Number of nodes in hidden layers.
            conditioner_depth (int): Number of hidden layers.
        """
        self.d = d
        self.D = D
        self.bijection = bijection
        output_size = self.bijection.num_params(D - d)
        self.conditioner = eqx.nn.MLP(
            d, output_size, conditioner_width, conditioner_depth, key=key
        )

    def __call__(self, x: jnp.ndarray):
        return self.transform_and_log_abs_det_jacobian(x)

    def transform_and_log_abs_det_jacobian(self, x):
        x_cond, x_trans = x[: self.d], x[self.d :]
        bijection_params = self.conditioner(x_cond)
        bijection_args = self.bijection.get_args(bijection_params)
        y_trans, log_abs_det = self.bijection.transform_and_log_abs_det_jacobian(
            x_trans, *bijection_args
        )
        y = jnp.concatenate([x_cond, y_trans])
        return y, log_abs_det

    def transform(self, x):
        x_cond, x_trans = x[: self.d], x[self.d :]
        bijection_params = self.conditioner(x_cond)
        bijection_args = self.bijection.get_args(bijection_params)
        y_trans = self.bijection.transform(x_trans, *bijection_args)
        y = jnp.concatenate([x_cond, y_trans])
        return y

    def inverse(self, y: jnp.ndarray):
        x_cond, y_trans = y[: self.d], y[self.d :]
        bijection_params = self.conditioner(x_cond)
        bijection_args = self.bijection.get_args(bijection_params)
        x_trans = self.bijection.inverse(y_trans, *bijection_args)
        x = jnp.concatenate([x_cond, x_trans])
        return x


class CouplingStack(Bijection, eqx.Module):
    layers: list
    D : int

    def __init__(
        self,
        key: random.PRNGKey,
        bijection: ParameterisedBijection,
        D: int,
        num_layers: int,
        conditioner_width: int = 40,
        conditioner_depth: int = 2,
    ):

        layers = []
        ds = [round(jnp.floor(D / 2).item()), round(jnp.ceil(D / 2).item())]
        permutation = jnp.flip(jnp.arange(D))
        for i in range(num_layers):
            key, subkey = random.split(key)
            d = ds[0] if i % 2 == 0 else ds[1]
            layers.extend(
                [
                    Coupling(
                        key=key, bijection=bijection, d=d, D=D,
                        conditioner_width=conditioner_width,
                        conditioner_depth=conditioner_depth
                    ),
                    Permute(permutation),
                ]
            )
        self.layers = layers[:-2]  # remove last leakyRelu and permute
        self.D = D

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
