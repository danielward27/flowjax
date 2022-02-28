from jaxflows.bijections.abc import ParameterisedBijection
import equinox as eqx
import jax.numpy as jnp
import jax
from jax import random
from jaxflows.bijections.permute import Permute
from jaxflows.bijections.abc import Bijection


class IgnoreCondition(Bijection):
    """Wrap bijection to allow it to take and ignore additional
    conditioning variables. Facilitates simple stacking of layers."""

    bijection: Bijection

    def __init__(self, bijection):
        self.bijection = bijection

    def transform(self, x, condition=None):
        return self.bijection.transform(x)

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        return self.bijection.transform_and_log_abs_det_jacobian(x)

    def inverse(self, y, condition=None):
        return self.bijection.inverse(y)


class Coupling(Bijection, eqx.Module):
    d: int
    D: int
    bijection: ParameterisedBijection
    conditioner: eqx.nn.MLP
    condition_dim: int

    def __init__(
        self,
        key: random.PRNGKey,
        bijection: ParameterisedBijection,
        d: int,
        D: int,
        nn_width: int,
        nn_depth: int,
        condition_dim: int = 0,
    ):
        """Coupling layer implementation.

        Args:
            key (random.PRNGKey): Key
            bijection (ParameterisedBijection): Bijection to be parameterised by the conditioner neural netork.
            d (int): Number of untransformed conditioning variables.
            D (int): Total dimension.
            nn_width (int): Number of nodes in hidden layers.
            nn_depth (int): Number of hidden layers.
        """
        self.d = d
        self.D = D
        self.bijection = bijection
        output_size = self.bijection.num_params(D - d)

        self.conditioner = eqx.nn.MLP(
            in_size=d + condition_dim,
            out_size=output_size,
            width_size=nn_width,
            depth=nn_depth,
            key=key,
        )

        self.condition_dim = condition_dim

    def transform(self, x, condition=None):
        if condition is None and self.condition_dim == 0:
            condition = jnp.empty((0,))

        x_cond, x_trans = x[: self.d], x[self.d :]
        cond = jnp.concatenate((x_cond, condition))
        bijection_params = self.conditioner(cond)
        bijection_args = self.bijection.get_args(bijection_params)
        y_trans = self.bijection.transform(x_trans, *bijection_args)
        y = jnp.concatenate((x_cond, y_trans))
        return y

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        if condition is None and self.condition_dim == 0:
            condition = jnp.empty((0,))

        x_cond, x_trans = x[: self.d], x[self.d :]
        cond = jnp.concatenate((x_cond, condition))

        bijection_params = self.conditioner(cond)
        bijection_args = self.bijection.get_args(bijection_params)
        y_trans, log_abs_det = self.bijection.transform_and_log_abs_det_jacobian(
            x_trans, *bijection_args
        )
        y = jnp.concatenate([x_cond, y_trans])
        return y, log_abs_det

    def inverse(self, y: jnp.ndarray, condition=None):
        if condition is None and self.condition_dim == 0:
            condition = jnp.empty((0,))

        x_cond, y_trans = y[: self.d], y[self.d :]
        cond = jnp.concatenate((x_cond, condition))
        bijection_params = self.conditioner(cond)
        bijection_args = self.bijection.get_args(bijection_params)
        x_trans = self.bijection.inverse(y_trans, *bijection_args)
        x = jnp.concatenate((x_cond, x_trans))
        return x


class CouplingStack(Bijection, eqx.Module):
    layers: list
    D: int

    def __init__(
        self,
        key: random.PRNGKey,
        bijection: ParameterisedBijection,
        D: int,
        num_layers: int,
        nn_width: int = 40,
        nn_depth: int = 2,
        condition_dim: int = 0,
    ):

        layers = []
        ds = [round(jnp.floor(D / 2).item()), round(jnp.ceil(D / 2).item())]
        permutation = jnp.flip(jnp.arange(D))
        for i in range(num_layers):
            key, coupling_key = random.split(key)
            d = ds[0] if i % 2 == 0 else ds[1]
            layers.extend(
                [
                    Coupling(
                        key=coupling_key,
                        bijection=bijection,
                        d=d,
                        D=D,
                        nn_width=nn_width,
                        nn_depth=nn_depth,
                        condition_dim=condition_dim,
                    ),
                    IgnoreCondition(Permute(permutation)),
                ]
            )
        self.layers = layers[
            :-1
        ]  # TODO check this works as expected (remove last permute)
        self.D = D

    def transform(self, x, condition=None):
        if condition is None and self.condition_dim == 0:
            condition = jnp.empty((0,))

        z = x
        for layer in self.layers:
            z = layer.transform(z, condition)
        return z

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        if condition is None and self.condition_dim == 0:
            condition = jnp.empty((0,))

        log_abs_det_jac = 0
        z = x
        for layer in self.layers:
            z, log_abs_det_jac_i = layer.transform_and_log_abs_det_jacobian(
                z, condition
            )
            log_abs_det_jac += log_abs_det_jac_i
        return z, log_abs_det_jac

    def inverse(self, z, condition=None):
        if condition is None and self.condition_dim == 0:
            condition = jnp.empty((0,))
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x, condition)
        return x
