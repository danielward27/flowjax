"""
Block Neural Autoregressive bijection implementation.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import random
from jax.random import KeyArray

from flowjax.bijections import Bijection
from flowjax.nn.block_autoregressive import (BlockAutoregressiveLinear,
                                             BlockTanh)


class BlockAutoregressiveNetwork(Bijection):
    """Block Autoregressive Network (https://arxiv.org/abs/1904.04676)."""
    dim: int
    depth: int
    layers: list
    cond_dim: int
    block_dim: int
    activation: Callable

    def __init__(
        self,
        key: KeyArray,
        dim: int,
        cond_dim: int,
        depth: int,
        block_dim: int,
        activation: Optional[Callable] = None,
    ):
        """
        Args:
            key (KeyArray): Jax PRNGKey
            dim (int): Dimension of the distribution.
            cond_dim (int): Dimension of extra conditioning variables.
            depth (int): Number of hidden layers in the network.
            block_dim (int): Block dimension (hidden layer size is `dim*block_dim`).
            activation (Callable, optional): Activation function. Defaults to BlockTanh.
        """
        activation = BlockTanh(dim) if activation is None else activation
        layers = []
        if depth == 0:
            layers.append(BlockAutoregressiveLinear(key, dim, (1, 1), cond_dim))
        else:
            keys = random.split(key, depth + 1)

            block_shapes = [
                (block_dim, 1),
                *[(block_dim, block_dim)] * (depth - 1),
                (1, block_dim),
            ]
            cond_dims = [cond_dim] + [0] * depth

            for key, block_shape, cd in zip(keys, block_shapes, cond_dims):

                layers.extend(
                    [
                        BlockAutoregressiveLinear(key, dim, block_shape, cd),
                        activation,
                    ]
                )
            layers = layers[:-1]  # remove last activation

        self.dim = dim
        self.depth = depth
        self.layers = layers
        self.cond_dim = cond_dim
        self.block_dim = block_dim
        self.activation = activation

    def transform(self, x, condition=None):
        x = self.layers[0](x, condition)[0]
        for layer in self.layers[1:]:
            x = layer(x)[0]
        return x

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        x, log_det_3d_0 = self.layers[0](x, condition)
        log_det_3ds = [log_det_3d_0]
        for layer in self.layers[1:]:
            x, log_det_3d = layer(x)
            log_det_3ds.append(log_det_3d)

        logdet = log_det_3ds[-1]
        for ld in reversed(log_det_3ds[:-1]):  # TODO avoid loop
            logdet = logmatmulexp(logdet, ld)
        return x, logdet.sum()

    def inverse(self, *args, **kwargs):
        raise NotImplementedError(
            "This transform would require numerical methods for inversion."
        )

    def inverse_and_log_abs_det_jacobian(self, *args, **kwargs):
        raise NotImplementedError(
            "This transform would require numerical methods for inversion."
        )


def logmatmulexp(x, y):
    """
    Numerically stable version of ``(x.log() @ y.log()).exp()``. From numpyro https://github.com/pyro-ppl/numpyro/blob/f2ff89a3a7147617e185eb51148eb15d56d44661/numpyro/distributions/util.py#L387
    """
    x_shift = jax.lax.stop_gradient(jnp.amax(x, -1, keepdims=True))
    y_shift = jax.lax.stop_gradient(jnp.amax(y, -2, keepdims=True))
    xy = jnp.log(jnp.matmul(jnp.exp(x - x_shift), jnp.exp(y - y_shift)))
    return xy + x_shift + y_shift
