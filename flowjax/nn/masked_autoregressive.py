"""Autoregressive linear layers and multilayer perceptron."""
from __future__ import annotations

from collections.abc import Callable

import jax.nn as jnn
import jax.numpy as jnp
from equinox import Module
from equinox.nn import Linear
from jax import Array, random
from jax.typing import ArrayLike

from flowjax.masks import rank_based_mask


def _identity(x):
    return x


class MaskedLinear(Module):
    """Masked linear neural network layer.

    Args:
        mask: Mask with shape (out_features, in_features).
        key: Jax random key.
        use_bias: Whether to include bias terms. Defaults to True.
    """

    linear: Linear
    mask: Array

    def __init__(self, mask: ArrayLike, *, use_bias: bool = True, key: Array):
        mask = jnp.asarray(mask)
        self.linear = Linear(mask.shape[1], mask.shape[0], use_bias, key=key)
        self.mask = mask

    def __call__(self, x: ArrayLike):
        """Run the masked linear layer.

        Args:
            x: Array with shape ``(mask.shape[1], )``
        """
        x = jnp.asarray(x)
        x = self.linear.weight * self.mask @ x
        if self.linear.bias is not None:
            x = x + self.linear.bias
        return x


class AutoregressiveMLP(Module):
    """An autoregressive multilayer perceptron.

    Similar to ``equinox.nn.composed.MLP``, however, connections will only exist between
    nodes where in_ranks < out_ranks.

    Args:
        in_ranks: Ranks of the inputs.
        hidden_ranks: Ranks of the hidden layer(s).
        out_ranks: Ranks of the outputs.
        depth: Number of hidden layers.
        activation: Activation function. Defaults to jnn.relu.
        final_activation: Final activation function. Defaults to _identity.
        key: Jax PRNGKey.
    """

    in_size: int
    out_size: int
    width_size: int
    depth: int
    in_ranks: Array
    out_ranks: Array
    hidden_ranks: Array
    layers: list[MaskedLinear]
    activation: Callable
    final_activation: Callable

    def __init__(
        self,
        in_ranks: ArrayLike,
        hidden_ranks: ArrayLike,
        out_ranks: ArrayLike,
        depth: int,
        activation: Callable = jnn.relu,
        final_activation: Callable = _identity,
        *,
        key,
    ) -> None:
        in_ranks, hidden_ranks, out_ranks = (
            jnp.asarray(a, jnp.int32) for a in (in_ranks, hidden_ranks, out_ranks)
        )
        masks = []
        if depth == 0:
            masks.append(rank_based_mask(in_ranks, out_ranks, eq=False))
        else:
            masks.append(rank_based_mask(in_ranks, hidden_ranks, eq=True))
            masks.extend(
                rank_based_mask(hidden_ranks, hidden_ranks, eq=True)
                for _ in range(depth - 1)
            )
            masks.append(rank_based_mask(hidden_ranks, out_ranks, eq=False))

        keys = random.split(key, len(masks))
        layers = tuple(
            MaskedLinear(mask, key=key) for mask, key in zip(masks, keys, strict=True)
        )

        self.layers = layers
        self.in_size = len(in_ranks)
        self.out_size = len(out_ranks)
        self.width_size = len(hidden_ranks)
        self.depth = depth
        self.in_ranks = in_ranks
        self.hidden_ranks = hidden_ranks
        self.out_ranks = out_ranks
        self.activation = activation
        self.final_activation = final_activation

    def __call__(self, x: Array):
        """Forward pass.

        Args:
            x: A JAX array with shape (in_size,).
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return self.final_activation(x)
