"""Masked autoregressive network and bijection."""

from functools import partial
from typing import Callable, Optional

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax.random import KeyArray

from flowjax.bijections import Bijection
from flowjax.transformers import Transformer
from flowjax.nn import AutoregressiveMLP
from flowjax.utils import Array, tile_until_length


class MaskedAutoregressive(Bijection):
    transformer: Transformer
    autoregressive_mlp: AutoregressiveMLP
    cond_dim: int

    def __init__(
        self,
        key: KeyArray,
        transformer: Transformer,
        dim: int,
        cond_dim: int,
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jnn.relu,
    ) -> None:
        """Masked autoregressive bijection implementation (https://arxiv.org/abs/1705.07057v4).
        The transformer is parameterised by a neural network, with weights masked to ensure
        an autoregressive structure.

        Args:
            key (KeyArray): Jax PRNGKey
            transformer (Transformer): Transformer to be parameterised by the autoregressive network.
            dim (int): Dimension.
            cond_dim (int): Dimension of any conditioning variables.
            nn_width (int): Neural network width.
            nn_depth (int): Neural network depth.
            nn_activation (Callable, optional): Neural network activation. Defaults to jnn.relu.
        """

        self.cond_dim = cond_dim

        in_ranks = jnp.concatenate(
            (jnp.arange(dim), -jnp.ones(cond_dim))
        )  # we give conditioning variables rank -1
        hidden_ranks = tile_until_length(jnp.arange(dim), nn_width)
        out_ranks = transformer.get_ranks(dim)
        self.transformer = transformer
        self.autoregressive_mlp = AutoregressiveMLP(
            in_ranks,
            hidden_ranks,
            out_ranks,
            nn_depth,
            nn_activation,
            key=key,
        )

    def transform(self, x, condition=None):
        transformer_args = self.get_transformer_args(x, condition)
        y = self.transformer.transform(x, *transformer_args)
        return y

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        transformer_args = self.get_transformer_args(x, condition)
        y, log_abs_det = self.transformer.transform_and_log_abs_det_jacobian(
            x, *transformer_args
        )
        return y, log_abs_det

    def inverse(self, y, condition=None):
        init = (y, 0)
        fn = partial(self.inv_scan_fn, condition=condition)
        (x, _), _ = jax.lax.scan(fn, init, None, length=len(y))
        return x

    def inv_scan_fn(self, init, _, condition):
        "One 'step' in computing the inverse"
        y, rank = init
        transformer_args = self.get_transformer_args(y, condition)
        x = self.transformer.inverse(y, *transformer_args)
        x = y.at[rank].set(x[rank])
        return (x, rank + 1), None

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        x = self.inverse(y, condition)
        log_det = self.transform_and_log_abs_det_jacobian(x, condition)[1]
        return x, -log_det

    def get_transformer_args(self, x: Array, condition: Optional[Array] = None):
        nn_input = x if condition is None else jnp.concatenate((x, condition))
        transformer_params = self.autoregressive_mlp(nn_input)
        transformer_args = self.transformer.get_args(transformer_params)
        return transformer_args
