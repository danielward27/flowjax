from typing import Callable, Union
import math
import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
from jax.random import KeyArray
from flowjax.bijections import Bijection
from flowjax.utils import get_ravelled_bijection_constructor, Array
from flowjax.bijections.jax_transforms import Vmap


class Coupling(Bijection):
    d: int
    D: int
    transformer_constructor: Callable
    conditioner: eqx.nn.MLP

    def __init__(
        self,
        key: KeyArray,
        transformer: Bijection,
        d: int,
        D: int,
        cond_dim: Union[None, int],
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jnn.relu,
    ):
        """Coupling layer implementation (https://arxiv.org/abs/1605.08803).
        Args:
            key (KeyArray): Jax PRNGKey
            transformer (Bijection): Bijection with shape () to be parameterised by the conditioner neural netork.
            d (int): Number of untransformed conditioning variables.
            D (int): Total dimension.
            cond_dim (Union[None, int]): Dimension of additional conditioning variables.
            nn_width (int): Neural network hidden layer width.
            nn_depth (int): Neural network hidden layer size.
            nn_activation (Callable, optional): Neural network activation function. Defaults to jnn.relu.
        """
        if transformer.shape != () or transformer.cond_shape is not None:
            raise ValueError(
                "Currently, only unconditional transformers with shape () are supported."
            )

        constructor, transformer_init_params = get_ravelled_bijection_constructor(
            transformer
        )

        self.transformer_constructor = constructor
        self.d = d
        self.D = D
        self.shape = (D,)
        self.cond_shape = (cond_dim,) if cond_dim is not None else None

        conditioner_output_size = transformer_init_params.size * (D - d)

        conditioner = eqx.nn.MLP(
            in_size=d if cond_dim is None else d + cond_dim,
            out_size=conditioner_output_size,
            width_size=nn_width,
            depth=nn_depth,
            activation=nn_activation,
            key=key,
        )

        # Initialise bias terms to match the provided transformer parameters
        self.conditioner = eqx.tree_at(
            where=lambda mlp: mlp.layers[-1].bias,
            pytree=conditioner,
            replace=jnp.tile(transformer_init_params, D - d),
        )

    def transform(self, x, condition=None):
        x_cond, x_trans = x[: self.d], x[self.d :]
        nn_input = x_cond if condition is None else jnp.hstack((x_cond, condition))
        transformer_params = self.conditioner(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        y_trans = transformer.transform(x_trans)
        y = jnp.hstack((x_cond, y_trans))
        return y

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        x_cond, x_trans = x[: self.d], x[self.d :]
        nn_input = x_cond if condition is None else jnp.hstack((x_cond, condition))
        transformer_params = self.conditioner(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        y_trans, log_det = transformer.transform_and_log_abs_det_jacobian(x_trans)
        y = jnp.hstack((x_cond, y_trans))
        return y, log_det

    def inverse(self, y, condition=None):
        x_cond, y_trans = y[: self.d], y[self.d :]
        nn_input = x_cond if condition is None else jnp.concatenate((x_cond, condition))
        transformer_params = self.conditioner(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        x_trans = transformer.inverse(y_trans)
        x = jnp.hstack((x_cond, x_trans))
        return x

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        x_cond, y_trans = y[: self.d], y[self.d :]
        nn_input = x_cond if condition is None else jnp.concatenate((x_cond, condition))
        transformer_params = self.conditioner(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        x_trans, log_det = transformer.inverse_and_log_abs_det_jacobian(y_trans)
        x = jnp.hstack((x_cond, x_trans))
        return x, log_det

    def _flat_params_to_transformer(
        self, params: Array
    ):  # TODO code repetition with MAF
        "Reshape to dim X params_per_dim, then vmap."
        dim = self.D - self.d
        transformer_params = jnp.reshape(params, (dim, -1))
        transformer = eqx.filter_vmap(self.transformer_constructor)(transformer_params)
        return Vmap(transformer, (dim,))
