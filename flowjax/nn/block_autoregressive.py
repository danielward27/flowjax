"""Block autoregressive neural network components."""
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import Array, random
from jax.nn.initializers import glorot_uniform
from jax.random import KeyArray

from flowjax.masks import block_diag_mask, block_tril_mask


class BlockAutoregressiveLinear(eqx.Module):
    """Block autoregressive neural network layer (https://arxiv.org/abs/1904.04676).

    Conditioning variables are incorporated by appending columns (one for each
    conditioning variable) to the right of the block diagonal weight matrix.

    Args:
        key: Random key
        n_blocks: Number of diagonal blocks (dimension of original input).
        block_shape: The shape of the (unconstrained) blocks.
        cond_dim: Number of additional conditioning variables. Defaults to None.
        init: Default initialisation method for the weight matrix. Defaults to
            ``glorot_uniform()``.
    """

    n_blocks: int
    block_shape: tuple
    cond_dim: int | None
    weights: Array
    bias: Array
    weight_log_scale: Array
    in_features: int
    out_features: int
    b_diag_mask: Array
    b_diag_mask_idxs: Array
    b_tril_mask: Array

    def __init__(
        self,
        key: KeyArray,
        *,
        n_blocks: int,
        block_shape: tuple,
        cond_dim: int | None = None,
        init: Callable | None = None,
    ):
        init = init if init is not None else glorot_uniform()
        self.cond_dim = cond_dim

        if cond_dim is None:
            cond_dim = 0

        cond_size = (block_shape[0] * n_blocks, cond_dim)

        self.b_diag_mask = jnp.column_stack(
            (block_diag_mask(block_shape, n_blocks), jnp.zeros(cond_size, jnp.int32)),
        )

        self.b_tril_mask = jnp.column_stack(
            (block_tril_mask(block_shape, n_blocks), jnp.ones(cond_size, jnp.int32)),
        )

        self.b_diag_mask_idxs = jnp.where(
            self.b_diag_mask,
            size=block_shape[0] * block_shape[1] * n_blocks,
        )

        in_features, out_features = (
            block_shape[1] * n_blocks + cond_dim,
            block_shape[0] * n_blocks,
        )

        *w_key, bias_key, scale_key = random.split(key, n_blocks + 2)

        self.weights = init(w_key[0], (out_features, in_features)) * (
            self.b_tril_mask + self.b_diag_mask
        )
        self.bias = (random.uniform(bias_key, (out_features,)) - 0.5) * (
            2 / jnp.sqrt(out_features)
        )

        self.n_blocks = n_blocks
        self.block_shape = block_shape
        self.cond_dim = cond_dim
        self.weight_log_scale = jnp.log(random.uniform(scale_key, (out_features, 1)))
        self.in_features = in_features
        self.out_features = out_features

    def get_normalised_weights(self):
        """Carries out weight normalisation."""
        weights = (
            jnp.exp(self.weights) * self.b_diag_mask + self.weights * self.b_tril_mask
        )
        weight_norms = jnp.linalg.norm(weights, axis=-1, keepdims=True)
        return jnp.exp(self.weight_log_scale) * weights / weight_norms

    def __call__(self, x, condition=None):
        """Returns output y, and components of weight matrix needed log_det component.

        The components of the weight matrix have shape
        ``(n_blocks, block_shape[0], block_shape[1])``.
        """
        weights = self.get_normalised_weights()
        if condition is not None:
            x = jnp.concatenate((x, condition))
        y = weights @ x + self.bias
        jac_3d = weights[self.b_diag_mask_idxs].reshape(
            self.n_blocks,
            *self.block_shape,
        )
        return y, jnp.log(jac_3d)
