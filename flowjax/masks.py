"""Masks used in flows.

Masks are generally used in flows to mask out some weights, in order to enforce a
dependency structure that ensures invertibility and efficient Jacobian determinant
calculations.
"""

import operator

import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import block_diag


def rank_based_mask(in_ranks: Array, out_ranks: Array, *, eq: bool = False):
    """Forms mask matrix, with 1s where the out_ranks > or >= in_ranks.

    Args:
        in_ranks: Ranks of the inputs.
        out_ranks: Ranks of the outputs.
        eq: If true, compares with >= instead of >. Defaults to False.

    Returns:
        Array: Mask with shape `(len(out_ranks), len(in_ranks))`
    """
    for ranks in (in_ranks, out_ranks):
        if ranks.ndim != 1:
            raise ValueError(f"Expected ranks.ndim==1, got {ranks.ndim}")
    op = operator.ge if eq else operator.gt
    return op(out_ranks[:, None], in_ranks).astype(jnp.int32)


def block_diag_mask(block_shape: tuple, n_blocks: int):
    """Block diagonal mask."""
    return block_diag(*jnp.ones((n_blocks, *block_shape), jnp.int32))


def block_tril_mask(block_shape: tuple, n_blocks: int):
    """Upper triangular block mask, excluding diagonal blocks."""
    mask = jnp.zeros((block_shape[0] * n_blocks, block_shape[1] * n_blocks), jnp.int32)
    for i in range(n_blocks):
        mask = mask.at[
            (i + 1) * block_shape[0] :,
            i * block_shape[1] : (i + 1) * block_shape[1],
        ].set(1)
    return mask
