from typing import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
from jax.random import KeyArray

from flowjax.bijections import Bijection
from flowjax.transformers import Transformer


class Coupling(Bijection):
    d: int
    D: int
    transformer: Transformer
    conditioner: eqx.nn.MLP
    cond_dim: int

    def __init__(
        self,
        key: KeyArray,
        transformer: Transformer,
        d: int,
        D: int,
        cond_dim: int,
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jnn.relu,
    ):
        """Coupling layer implementation (https://arxiv.org/abs/1605.08803).

        Args:
            key (KeyArray): Jax PRNGKey
            transformer (Transformer): Transformer to be parameterised by the conditioner neural netork.
            d (int): Number of untransformed conditioning variables.
            D (int): Total dimension.
            cond_dim (int): Dimension of additional conditioning variables.
            nn_width (int): Neural network hidden layer width.
            nn_depth (int): Neural network hidden layer size.
            nn_activation (Callable, optional): Neural network activation function. Defaults to jnn.relu.
        """

        self.transformer = transformer
        self.d = d
        self.D = D
        self.cond_dim = cond_dim

        output_size = self.transformer.num_params(D - d)

        self.conditioner = eqx.nn.MLP(
            in_size=d + cond_dim,
            out_size=output_size,
            width_size=nn_width,
            depth=nn_depth,
            activation=nn_activation,
            key=key,
        )

    def transform(self, x, condition=None):
        x_cond, x_trans = x[: self.d], x[self.d :]
        nn_input = x_cond if condition is None else jnp.concatenate((x_cond, condition))
        bijection_params = self.conditioner(nn_input)
        bijection_args = self.transformer.get_args(bijection_params)
        y_trans = self.transformer.transform(x_trans, *bijection_args)
        y = jnp.concatenate((x_cond, y_trans))
        return y

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        x_cond, x_trans = x[: self.d], x[self.d :]
        nn_input = x_cond if condition is None else jnp.concatenate((x_cond, condition))
        bijection_params = self.conditioner(nn_input)
        bijection_args = self.transformer.get_args(bijection_params)
        y_trans, log_abs_det = self.transformer.transform_and_log_abs_det_jacobian(
            x_trans, *bijection_args
        )
        y = jnp.concatenate([x_cond, y_trans])
        return y, log_abs_det

    def inverse(self, y, condition=None):
        x_cond, y_trans = y[: self.d], y[self.d :]
        nn_input = x_cond if condition is None else jnp.concatenate((x_cond, condition))
        bijection_params = self.conditioner(nn_input)
        bijection_args = self.transformer.get_args(bijection_params)
        x_trans = self.transformer.inverse(y_trans, *bijection_args)
        x = jnp.concatenate((x_cond, x_trans))
        return x

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        x_cond, y_trans = y[: self.d], y[self.d :]
        nn_input = x_cond if condition is None else jnp.concatenate((x_cond, condition))
        bijection_params = self.conditioner(nn_input)
        bijection_args = self.transformer.get_args(bijection_params)
        x_trans, log_det = self.transformer.inverse_and_log_abs_det_jacobian(
            y_trans, *bijection_args
        )
        x = jnp.concatenate((x_cond, x_trans))
        return x, log_det
