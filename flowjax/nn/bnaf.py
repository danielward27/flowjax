import jax.numpy as jnp
from jax import random
from jax.random import KeyArray
from jax.nn.initializers import glorot_uniform
from flowjax.utils import Array
import equinox as eqx
from typing import Callable
import jax
from flowjax.masks import block_diag_mask, block_tril_mask


class BlockAutoregressiveLinear(eqx.Module):
    n_blocks: int
    block_shape: tuple
    cond_dim: int
    W: Array
    bias: Array
    W_log_scale: Array
    in_features: int
    out_features: int
    b_diag_mask: Array
    b_diag_mask_idxs: Array
    b_tril_mask: Array

    def __init__(
        self,
        key: KeyArray,
        n_blocks: int,
        block_shape: tuple,
        cond_dim: int = 0,
        init: Callable = glorot_uniform(),
    ):
        """Block autoregressive neural network layer (https://arxiv.org/abs/1904.04676).
        Conditioning variables are incorporated by appending columns (one for each
        conditioning variable) to the left of the block diagonal weight matrix.

        Args:
            key KeyArray: Random key
            n_blocks (int): Number of diagonal blocks (dimension of original input).
            block_shape (tuple): The shape of the (unconstrained) blocks.
            cond_dim (int): Number of additional conditioning variables. Defaults to 0.
            init (Callable, optional): Default initialisation method for the weight matrix. Defaults to glorot_uniform().
        """
        cond_size = (block_shape[0] * n_blocks, cond_dim)

        self.b_diag_mask = jnp.column_stack(
            (jnp.zeros(cond_size, jnp.int32), block_diag_mask(block_shape, n_blocks))
        )

        self.b_tril_mask = jnp.column_stack(
            (jnp.ones(cond_size, jnp.int32), block_tril_mask(block_shape, n_blocks))
        )
        self.b_diag_mask_idxs = jnp.where(self.b_diag_mask)

        in_features, out_features = (
            block_shape[1] * n_blocks + cond_dim,
            block_shape[0] * n_blocks,
        )

        *w_key, bias_key, scale_key = random.split(key, n_blocks + 2)

        self.W = init(w_key[0], (out_features, in_features)) * (
            self.b_tril_mask + self.b_diag_mask
        )
        self.bias = (random.uniform(bias_key, (out_features,)) - 0.5) * (
            2 / jnp.sqrt(out_features)
        )

        self.n_blocks = n_blocks
        self.block_shape = block_shape
        self.cond_dim = cond_dim
        self.W_log_scale = jnp.log(random.uniform(scale_key, (out_features, 1)))
        self.in_features = in_features
        self.out_features = out_features

    def get_normalised_weights(self):
        "Carries out weight normalisation."
        W = jnp.exp(self.W) * self.b_diag_mask + self.W * self.b_tril_mask
        W_norms = jnp.linalg.norm(W, axis=-1, keepdims=True)
        return jnp.exp(self.W_log_scale) * W / W_norms

    def __call__(self, x, condition=None):
        "returns output y, and components of weight matrix needed log_det component (n_blocks, block_shape[0], block_shape[1])"
        W = self.get_normalised_weights()
        if condition is not None:
            x = jnp.concatenate((condition, x))
        y = W @ x + self.bias
        jac_3d = W[self.b_diag_mask_idxs].reshape(self.n_blocks, *self.block_shape)
        return y, jnp.log(jac_3d)
