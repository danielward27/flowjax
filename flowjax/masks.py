"""
Various masks, generally used in flows to enforce e.g. a dependency structure
that leads to invertibility and efficient Jacobian determinant calculations.
"""

import jax.numpy as jnp
from jax.scipy.linalg import block_diag

from flowjax.utils import Array


def rank_based_mask(in_ranks: Array, out_ranks: Array, eq: bool = False):
    """Forms mask matrix, with 1s where the out_ranks > or >= in_ranks.

    Args:
        in_ranks (Array): Ranks of the inputs.
        out_ranks (Array): Ranks of the outputs.
        eq (bool): If true, compares with >= instead of >. Defaults to False.

    Returns:
        Array: Mask with shape `(len(out_ranks), len(in_ranks))`
    """

    assert (in_ranks.ndim) == 1 and (out_ranks.ndim == 1)
    if eq:
        mask = out_ranks[:, None] >= in_ranks
    else:
        mask = out_ranks[:, None] > in_ranks
    return mask.astype(jnp.int32)


def block_diag_mask(block_shape: tuple, n_blocks: int):
    "Block diagonal mask."
    return block_diag(*jnp.ones((n_blocks, *block_shape), jnp.int32))


def block_tril_mask(block_shape: tuple, n_blocks: int):
    "Upper triangular block mask, excluding diagonal blocks."
    mask = jnp.zeros((block_shape[0] * n_blocks, block_shape[1] * n_blocks), jnp.int32)
    for i in range(n_blocks):
        mask = mask.at[
            (i + 1) * block_shape[0] :, i * block_shape[1] : (i + 1) * block_shape[1]
        ].set(1)
    return mask
