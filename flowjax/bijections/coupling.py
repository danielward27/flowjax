from flowjax.bijections.abc import ParameterisedBijection
import equinox as eqx
import jax.numpy as jnp
from flowjax.bijections.abc import Bijection
from jax.random import KeyArray
from flowjax.utils import Array


class Coupling(Bijection):
    d: int
    D: int
    bijection: ParameterisedBijection
    conditioner: eqx.nn.MLP
    cond_dim: int

    def __init__(
        self,
        key: KeyArray,
        bijection: ParameterisedBijection,
        d: int,
        D: int,
        cond_dim: int,
        nn_width: int,
        nn_depth: int,
    ):
        """Coupling layer implementation.

        Args:
            key KeyArray: Key
            bijection (ParameterisedBijection): Bijection to be parameterised by the conditioner neural netork.
            d (int): Number of untransformed conditioning variables.
            D (int): Total dimension.
            nn_width (int): Number of nodes in hidden layers.
            nn_depth (int): Number of hidden layers.
        """
        self.bijection = bijection
        self.d = d
        self.D = D
        self.cond_dim = cond_dim

        output_size = self.bijection.num_params(D - d)

        self.conditioner = eqx.nn.MLP(
            in_size=d + cond_dim,
            out_size=output_size,
            width_size=nn_width,
            depth=nn_depth,
            key=key,
        )

    def transform(self, x: Array, condition=None):
        x_cond, x_trans = x[: self.d], x[self.d :]
        nn_input = x_cond if condition is None else jnp.concatenate((x_cond, condition))
        bijection_params = self.conditioner(nn_input)
        bijection_args = self.bijection.get_args(bijection_params)
        y_trans = self.bijection.transform(x_trans, *bijection_args)
        y = jnp.concatenate((x_cond, y_trans))
        return y

    def transform_and_log_abs_det_jacobian(self, x: Array, condition=None):
        x_cond, x_trans = x[: self.d], x[self.d :]
        nn_input = x_cond if condition is None else jnp.concatenate((x_cond, condition))
        bijection_params = self.conditioner(nn_input)
        bijection_args = self.bijection.get_args(bijection_params)
        y_trans, log_abs_det = self.bijection.transform_and_log_abs_det_jacobian(
            x_trans, *bijection_args
        )
        y = jnp.concatenate([x_cond, y_trans])
        return y, log_abs_det

    def inverse(self, y: Array, condition=None):
        x_cond, y_trans = y[: self.d], y[self.d :]
        nn_input = x_cond if condition is None else jnp.concatenate((x_cond, condition))
        bijection_params = self.conditioner(nn_input)
        bijection_args = self.bijection.get_args(bijection_params)
        x_trans = self.bijection.inverse(y_trans, *bijection_args)
        x = jnp.concatenate((x_cond, x_trans))
        return x
