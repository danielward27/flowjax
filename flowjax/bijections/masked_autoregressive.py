"""Masked autoregressive network and bijection."""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray

from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.jax_transforms import Vmap
from flowjax.masks import rank_based_mask
from flowjax.utils import get_ravelled_pytree_constructor
from flowjax.wrappers import Parameterize


class MaskedAutoregressive(AbstractBijection):
    """Masked autoregressive bijection.

    The transformer is parameterised by a neural network, with weights masked to ensure
    an autoregressive structure.

    Refs:
        - https://arxiv.org/abs/1705.07057v4
        - https://arxiv.org/abs/1705.07057v4

    Args:
        key: Jax key
        transformer: Bijection with shape () to be parameterised by the autoregressive
            network. Parameters wrapped with ``NonTrainable`` are exluded.
        dim: Dimension.
        cond_dim: Dimension of any conditioning variables. Defaults to None.
        nn_width: Neural network width.
        nn_depth: Neural network depth.
        nn_activation: Neural network activation. Defaults to jnn.relu.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    transformer_constructor: Callable
    masked_autoregressive_mlp: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        transformer: AbstractBijection,
        dim: int,
        cond_dim: int | None = None,
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jnn.relu,
    ) -> None:
        if transformer.shape != () or transformer.cond_shape is not None:
            raise ValueError(
                "Only unconditional transformers with shape () are supported.",
            )

        constructor, num_params = get_ravelled_pytree_constructor(transformer)

        if cond_dim is None:
            self.cond_shape = None
            in_ranks = jnp.arange(dim)
            # If dim=1, hidden ranks all zero -> all weights masked out in final layer
            hidden_ranks = jnp.arange(nn_width) % (dim - 1)
        else:
            self.cond_shape = (cond_dim,)
            # we give conditioning variables rank -1 (no masking of edges to output)
            in_ranks = jnp.hstack((jnp.arange(dim), -jnp.ones(cond_dim, int)))
            # If dim=1, hidden ranks all -1 -> all outputs only depend on condition
            hidden_ranks = (jnp.arange(nn_width) % dim) - 1
        out_ranks = jnp.repeat(jnp.arange(dim), num_params)

        self.masked_autoregressive_mlp = masked_autoregressive_mlp(
            in_ranks,
            hidden_ranks,
            out_ranks,
            depth=nn_depth,
            activation=nn_activation,
            key=key,
        )

        self.transformer_constructor = constructor
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)

    def transform(self, x, condition=None):
        nn_input = x if condition is None else jnp.hstack((x, condition))
        transformer_params = self.masked_autoregressive_mlp(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        return transformer.transform(x)

    def transform_and_log_det(self, x, condition=None):
        nn_input = x if condition is None else jnp.hstack((x, condition))
        transformer_params = self.masked_autoregressive_mlp(nn_input)
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
        transformer_params = self.masked_autoregressive_mlp(nn_input)
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
        return Vmap(transformer, in_axes=eqx.if_array(0))


def masked_autoregressive_mlp(
    in_ranks: Int[Array, " in_size"],
    hidden_ranks: Int[Array, " hidden_size"],
    out_ranks: Int[Array, " out_size"],
    **kwargs,
) -> eqx.nn.MLP:
    """Returns an equinox multilayer perceptron, with autoregressive masks.

    The weight matrices are wrapped using :class:`~flowjax.wrappers.Parameterize`, which
    will apply the masking when :class:`~flowjax.wrappers.unwrap` is called on the MLP.
    For details of how the masks are formed, see https://arxiv.org/pdf/1502.03509.pdf.

    Args:
        in_ranks: The ranks of the inputs.
        hidden_ranks: The ranks of the hidden dimensions.
        out_ranks: The ranks of the output dimensions.
        **kwargs: Keyword arguments passed to equinox.nn.MLP.
    """
    in_ranks, hidden_ranks, out_ranks = (
        jnp.asarray(a, jnp.int32) for a in (in_ranks, hidden_ranks, out_ranks)
    )  # TODO remove if using beartype
    mlp = eqx.nn.MLP(
        in_size=len(in_ranks),
        out_size=len(out_ranks),
        width_size=len(hidden_ranks),
        **kwargs,
    )
    ranks = [in_ranks, *[hidden_ranks] * mlp.depth, out_ranks]

    masked_layers = []
    for i, linear in enumerate(mlp.layers):
        mask = rank_based_mask(ranks[i], ranks[i + 1], eq=i != len(mlp.layers) - 1)
        masked_linear = eqx.tree_at(
            lambda linear: linear.weight,
            linear,
            Parameterize(jnp.where, mask, linear.weight, 0),
        )
        masked_layers.append(masked_linear)

    return eqx.tree_at(lambda mlp: mlp.layers, mlp, replace=tuple(masked_layers))
