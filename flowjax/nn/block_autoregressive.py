"""Block autoregressive neural network components."""

from math import prod

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from flowjax import masks
from flowjax.bijections.softplus import SoftPlus
from flowjax.wrappers import BijectionReparam, WeightNormalization, Where


class BlockAutoregressiveLinear(eqx.Module):
    """Block autoregressive linear layer (https://arxiv.org/abs/1904.04676).

    Conditioning variables are incorporated by appending columns (one for each
    conditioning variable) to the right of the block diagonal weight matrix.

    Args:
        key: Random key.
        n_blocks: Number of diagonal blocks (dimension of original input).
        block_shape: The shape of the blocks.
        cond_dim: Number of additional conditioning variables. Defaults to None.
    """

    linear: eqx.nn.Linear
    cond_dim: int | None
    block_shape: tuple[int, ...]
    n_blocks: int
    block_diag_idxs: Array

    def __init__(
        self,
        key: Array,
        *,
        n_blocks: int,
        block_shape: tuple,
        cond_dim: int | None = None,
    ):
        self.cond_dim = cond_dim
        cond_dim = 0 if cond_dim is None else cond_dim

        in_features = block_shape[1] * n_blocks + cond_dim
        out_features = block_shape[0] * n_blocks
        linear = eqx.nn.Linear(in_features, out_features, key=key)

        def _right_pad(arr, val, width):
            return jnp.column_stack((arr, jnp.full((arr.shape[0], width), val, int)))

        block_diag_mask = _right_pad(
            masks.block_diag_mask(block_shape, n_blocks).astype(bool),
            val=False,
            width=cond_dim,
        )

        block_tril_mask = _right_pad(
            masks.block_tril_mask(block_shape, n_blocks).astype(
                bool
            ),  # TODO masks to bools?
            val=True,
            width=cond_dim,
        )

        weight = Where(block_tril_mask, linear.weight, 0)
        weight = Where(
            block_diag_mask,
            BijectionReparam(weight, SoftPlus(), invert_on_init=False),
            weight,
        )
        weight = WeightNormalization(weight)

        self.linear = eqx.tree_at(lambda linear: linear.weight, linear, replace=weight)
        self.block_shape = block_shape
        self.n_blocks = n_blocks

        self.block_diag_idxs = jnp.where(
            block_diag_mask, size=prod(block_shape) * n_blocks
        )

    def __call__(self, x, condition=None):
        """Returns output y, and components of weight matrix needed log_det computation.

        The weights of this module need to be unwrapped before use if using directly.
        The log det term has shape``(n_blocks, *block_shape)``.
        """
        if condition is not None:
            x = jnp.concatenate((x, condition))
        y = self.linear(x)
        jac_3d = self.linear.weight[self.block_diag_idxs].reshape(
            self.n_blocks, *self.block_shape
        )
        return y, jnp.log(jac_3d)
