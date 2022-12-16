"""
Various masks, generally used in flows to enforce e.g. a dependency structure
that leads to invertibility and efficient Jacobian determinant calculations.
"""

from flowjax.utils import Array
import jax.numpy as jnp
from jax.scipy.linalg import block_diag


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


# %%


# def block_diag_mask(block_shape, n_blocks):
#     height, width = block_shape
#     out = jnp.zeros((n_blocks * height, n_blocks * width), jnp.int32)

#     def single_block(jnp.array(i))

#     # def single_block(carry, _):
#     #     (arr, i) = carry
#     #     arr = arr.at[
#     #         (i * height) : ((i + 1) * height), (i * width) : ((i + 1) * width)
#     #     ].set(1)
#     #     return (arr, i + 1), None

#     # return scan(single_block, (out, 0), None, length=n_blocks)

#     for i in range(n_blocks):
#         out = out.at[
#             (i * height) : ((i + 1) * height), (i * width) : ((i + 1) * width)
#         ].set(1)

#     return out


# %%
# from jax.scipy.linalg import block_diag
# import equinox as eqx
# import jax.numpy as jnp

# block_diag(jnp.array([jnp.ones((3,3)) for i in range(3)]))

# %%

# def outer(array, block_shape, n_blocks):
#     return block_diag_mask(block_shape, n_blocks)

# def block_diag_mask(block_shape: tuple, n_blocks: int):
#     "Block diagonal mask."
#     return block_diag(*jnp.ones((n_blocks, *block_shape), jnp.int32))


# eqx.filter_vmap(outer)(jnp.ones(3), (5,3), 3).shape


# %%
