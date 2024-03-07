"""Autoregressive linear layers and multilayer perceptron."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike

from flowjax.masks import rank_based_mask
from flowjax.wrappers import Where


def autoregressive_mlp(
    in_ranks: ArrayLike,
    hidden_ranks: ArrayLike,
    out_ranks: ArrayLike,
    **kwargs,
) -> eqx.nn.MLP:  # TODO document
    """Returns an equinox multilayer perceptron, with autoregressive masks.

    The weight matrices are wrapped using :class:`~flowjax.wrappers.Where`, which
    will apply the masking when :class:`~flowjax.wrappers.unwrap` is called on the MLP.
    For details of how the masks are formed, see https://arxiv.org/pdf/1502.03509.pdf.

    Args:
        in_ranks: The ranks of the inputs.
        hidden_ranks: The ranks of the hidden dimensions.
        out_ranks: The ranks of the output dimensions.

    """
    in_ranks, hidden_ranks, out_ranks = (
        jnp.asarray(a, jnp.int32) for a in (in_ranks, hidden_ranks, out_ranks)
    )
    mlp = eqx.nn.MLP(
        in_size=len(in_ranks),
        out_size=len(out_ranks),
        width_size=len(hidden_ranks),
        **kwargs,
    )
    ranks = [in_ranks, *[hidden_ranks] * mlp.depth, out_ranks]

    masked_layers = []
    for i, linear in enumerate(mlp.layers):
        mask = rank_based_mask(ranks[i], ranks[i + 1], eq=i == len(mlp.layers) - 1)
        masked_linear = eqx.tree_at(
            lambda linear: linear.weight, linear, Where(mask, linear.weight, 0)
        )
        masked_layers.append(masked_linear)

    return eqx.tree_at(lambda mlp: mlp.layers, mlp, replace=tuple(masked_layers))
