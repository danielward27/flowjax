"""Masks used in flows.

Masks are generally used in flows to mask out some weights, in order to enforce a
dependency structure that ensures invertibility and efficient Jacobian determinant
calculations.
"""

import operator

import jax.numpy as jnp
from jaxtyping import Array, Bool, Int


def rank_based_mask(
    in_ranks: Int[Array, " a"],
    out_ranks: Int[Array, " b"],
    *,
    eq: bool = False,
) -> Bool[Array, "b a"]:
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
    return op(out_ranks[:, None], in_ranks)


def block_diag_mask(block_shape: tuple, n_blocks: int) -> Bool[Array, "dim1 dim2"]:
    """Block diagonal mask."""
    mask = jnp.diag(jnp.ones(n_blocks, dtype=int))
    block = jnp.ones(block_shape, dtype=int)
    return jnp.kron(mask, block).astype(bool)


def block_tril_mask(
    block_shape: tuple, n_blocks: int, k: int = 0
) -> Bool[Array, "dim1 dim2"]:
    """Lower triangular block mask, with offset k."""
    mask = jnp.tri(n_blocks, n_blocks, k=k, dtype=int)
    block = jnp.ones(block_shape, dtype=int)
    return jnp.kron(mask, block).astype(bool)
