from typing import Callable, List

import jax.nn as jnn
import jax.numpy as jnp
from equinox import Module
from equinox.nn import Linear
from jax import random
from jax.random import KeyArray

from flowjax.masks import rank_based_mask
from flowjax.utils import Array, _identity


class MaskedLinear(Module):
    linear: Linear
    mask: Array

    def __init__(self, mask: Array, use_bias: bool = True, *, key: KeyArray):
        """
        Masked linear layer.

        Args:
            mask (Array): Mask with shape (out_features, in_features).
            key (KeyArray): Jax PRNGKey
            use_bias (bool, optional): Whether to include bias terms. Defaults to True.
        """
        self.linear = Linear(mask.shape[1], mask.shape[0], use_bias, key=key)
        self.mask = mask

    def __call__(self, x: Array):
        x = self.linear.weight * self.mask @ x
        if self.linear.bias is not None:
            x = x + self.linear.bias
        return x


class AutoregressiveMLP(Module):
    in_size: int
    out_size: int
    width_size: int
    depth: int
    in_ranks: Array
    out_ranks: Array
    hidden_ranks: Array
    layers: List[MaskedLinear]
    activation: Callable
    final_activation: Callable

    def __init__(
        self,
        in_ranks: Array,
        hidden_ranks: Array,
        out_ranks: Array,
        depth: int,
        activation: Callable = jnn.relu,
        final_activation: Callable = _identity,
        *,
        key
    ) -> None:
        """An autoregressive multilayer perceptron, similar to ``equinox.nn.composed.MLP``.
        Connections will only exist where in_ranks < out_ranks.

        Args:
            in_ranks (Array): Ranks of the inputs.
            hidden_ranks (Array): Ranks of the hidden layer(s).
            out_ranks (Array): Ranks of the outputs.
            depth (int): Number of hidden layers.
            activation (Callable, optional): Activation function. Defaults to jnn.relu.
            final_activation (Callable, optional): Final activation function. Defaults to _identity.
            key (KeyArray): Jax PRNGKey
        """

        masks = []
        if depth == 0:
            masks.append(rank_based_mask(in_ranks, out_ranks, eq=False))
        else:
            masks.append(rank_based_mask(in_ranks, hidden_ranks, eq=True))
            for _ in range(depth - 1):
                masks.append(rank_based_mask(hidden_ranks, hidden_ranks, eq=True))
            masks.append(rank_based_mask(hidden_ranks, out_ranks, eq=False))

        keys = random.split(key, len(masks))
        layers = [MaskedLinear(mask, key=key) for mask, key in zip(masks, keys)]

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
        x = self.final_activation(x)
        return x
