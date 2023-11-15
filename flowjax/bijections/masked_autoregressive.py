"""Masked autoregressive network and bijection."""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import Array
from jax.random import KeyArray

from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.jax_transforms import Vmap
from flowjax.nn import AutoregressiveMLP
from flowjax.utils import get_ravelled_bijection_constructor


class MaskedAutoregressive(AbstractBijection):
    """Masked autoregressive bijection.

    The transformer is parameterised by a neural network, with weights masked to ensure
    an autoregressive structure.

    Refs:
        - https://arxiv.org/abs/1705.07057v4
        - https://arxiv.org/abs/1705.07057v4

    Args:
        key: Jax PRNGKey
        transformer: Bijection with shape () to be parameterised by the autoregressive
            network.
        dim: Dimension.
        cond_dim: Dimension of any conditioning variables.
        nn_width: Neural network width.
        nn_depth: Neural network depth.
        nn_activation: Neural network activation. Defaults to jnn.relu.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    transformer_constructor: Callable
    autoregressive_mlp: AutoregressiveMLP

    def __init__(
        self,
        key: KeyArray,
        *,
        transformer: AbstractBijection,
        dim: int,
        cond_dim: int | None,
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jnn.relu,
    ) -> None:
        if transformer.shape != () or transformer.cond_shape is not None:
            raise ValueError(
                "Only unconditional transformers with shape () are supported.",
            )

        constructor, transformer_init_params = get_ravelled_bijection_constructor(
            transformer,
        )

        if cond_dim is None:
            self.cond_shape = None
            in_ranks = jnp.arange(dim)
        else:
            self.cond_shape = (cond_dim,)
            # we give conditioning variables rank -1 (no masking of edges to output)
            in_ranks = jnp.hstack(
                (jnp.arange(dim), -jnp.ones(cond_dim, dtype=jnp.int32)),
            )

        hidden_ranks = jnp.arange(nn_width) % dim
        out_ranks = jnp.repeat(jnp.arange(dim), transformer_init_params.size)

        autoregressive_mlp = AutoregressiveMLP(
            in_ranks,
            hidden_ranks,
            out_ranks,
            nn_depth,
            nn_activation,
            key=key,
        )

        # Initialise bias terms to match the provided transformer parameters
        self.autoregressive_mlp = eqx.tree_at(
            where=lambda t: t.layers[-1].linear.bias,
            pytree=autoregressive_mlp,
            replace=jnp.tile(transformer_init_params, dim),
        )

        self.transformer_constructor = constructor
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)

    def transform(self, x, condition=None):
        nn_input = x if condition is None else jnp.hstack((x, condition))
        transformer_params = self.autoregressive_mlp(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        return transformer.transform(x)

    def transform_and_log_det(self, x, condition=None):
        nn_input = x if condition is None else jnp.hstack((x, condition))
        transformer_params = self.autoregressive_mlp(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        return transformer.transform_and_log_det(x)

    def inverse(self, y, condition=None):
        init = (y, 0)
        fn = partial(self.inv_scan_fn, condition=condition)
        (x, _), _ = jax.lax.scan(fn, init, None, length=len(y))
        return x

    def inv_scan_fn(self, init, _, condition):
        """One 'step' in computing the inverse."""
        y, rank = init
        nn_input = y if condition is None else jnp.hstack((y, condition))
        transformer_params = self.autoregressive_mlp(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        x = transformer.inverse(y)
        x = y.at[rank].set(x[rank])
        return (x, rank + 1), None

    def inverse_and_log_det(self, y, condition=None):
        x = self.inverse(y, condition)
        log_det = self.transform_and_log_det(x, condition)[1]
        return x, -log_det

    def _flat_params_to_transformer(self, params: Array):
        """Reshape to dim X params_per_dim, then vmap."""
        dim = self.shape[-1]
        transformer_params = jnp.reshape(params, (dim, -1))
        transformer = eqx.filter_vmap(self.transformer_constructor)(transformer_params)
        return Vmap(transformer, in_axis=eqx.if_array(0))
