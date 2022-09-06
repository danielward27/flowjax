from typing import Callable, Optional
from equinox import Module
from equinox.nn import Linear
from jax import random
from jax.random import KeyArray
import jax.numpy as jnp
from flowjax.utils import tile_until_length
import jax.nn as jnn
from flowjax.bijections.abc import Bijection, ParameterisedBijection
from typing import List
from flowjax.utils import Array


def rank_based_mask(in_ranks, out_ranks, eq=False):
    """Mask with shape `(len(out_ranks), len(in_ranks))`, with 1s where the
    out_ranks > or >= in_ranks. If eq=True, then >= is used for comparison
    (i.e. allows connections between equal ranks)."""
    assert (in_ranks.ndim) == 1 and (out_ranks.ndim == 1)
    if eq:
        mask = out_ranks[:, None] >= in_ranks
    else:
        mask = out_ranks[:, None] > in_ranks
    return mask.astype(jnp.int32)


class MaskedLinear(Module):
    linear: Linear
    mask: Array

    def __init__(self, mask: Array, use_bias: bool = True, *, key):
        "Mask should have shape (out_features, in_features)."
        self.linear = Linear(mask.shape[1], mask.shape[0], use_bias, key=key)
        self.mask = mask

    def __call__(self, x: Array):
        x = self.linear.weight * self.mask @ x
        if self.linear.bias is not None:
            x = x + self.linear.bias
        return x


def _identity(x):
    return x


class AutoregressiveMLP(Module):
    in_size: int
    out_size: int
    width_size: int
    in_ranks: Array
    out_ranks: Array
    hidden_ranks: Array
    layers: List[MaskedLinear]
    depth: int
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
        """An autoregressive multilayer perceptron, similar to equinox.nn.composed.MLP.
        out_ranks controls the dependencies - paths will only exist between inputs and
        outputs if the input index is less than output rank.

        Args:
            in_size (int): Input dimension.
            out_ranks (Array): Ranks of the output. Connections will only exist where the input index < out_ranks.
            width_size (int): Hidden layer dimension.
            depth (int): Number of layers.
            key (jax.random.PRNGKey): Jax random key.
            activation (Callable, optional): Activation function. Defaults to jnn.relu.
            final_activation (Callable, optional): Final activation function. Defaults to _identity.
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
        self.in_ranks = in_ranks
        self.hidden_ranks = hidden_ranks
        self.out_ranks = out_ranks
        self.depth = depth
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


class MaskedAutoregressive(Bijection):
    bijection: ParameterisedBijection
    autoregressive_mlp: AutoregressiveMLP
    cond_dim: int

    def __init__(
        self,
        key: KeyArray,
        bijection: ParameterisedBijection,
        dim: int,
        cond_dim: int,
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jnn.relu,
    ) -> None:

        self.cond_dim = cond_dim

        out_ranks = bijection.get_ranks(dim)
        self.bijection = bijection

        in_ranks = jnp.concatenate(
            (jnp.arange(dim), -jnp.ones(cond_dim))
        )  # conditioning variables rank -1
        hidden_ranks = tile_until_length(jnp.arange(dim), nn_width)

        self.autoregressive_mlp = AutoregressiveMLP(
            in_ranks, hidden_ranks, out_ranks, nn_depth, nn_activation, key=key,
        )

    def transform(self, x, condition=None):
        if condition is not None:
            x = jnp.concatenate((x, condition))
        x_condition = jnp.concatenate((x, condition))
        bijection_params = self.autoregressive_mlp(x_condition)
        bijection_args = self.bijection.get_args(bijection_params)
        y = self.bijection.transform(x, *bijection_args)
        return y

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        if condition is not None:
            x = jnp.concatenate((x, condition))
        bijection_params = self.autoregressive_mlp(x)
        bijection_args = self.bijection.get_args(bijection_params)
        y, log_abs_det = self.bijection.transform_and_log_abs_det_jacobian(
            x, *bijection_args
        )
        return y, log_abs_det

    def inverse(self, y: Array, condition: Optional[Array] = None):
        raise NotImplementedError("Not yet implemented")
