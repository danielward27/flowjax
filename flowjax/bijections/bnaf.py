from typing import Tuple
import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.bijections.abc import Bijection

# Some adapted from https://github.com/pyro-ppl/numpyro/blob/master/numpyro/nn/block_neural_arn.py


class BlockMaskedDense(eqx.Module, Bijection):
    num_blocks: int
    block_shape: tuple
    block_diag_mask: jnp.ndarray
    block_tril_mask: jnp.ndarray

    def __init__(self, num_blocks: int, block_shape: tuple):
        self.block_shape = block_shape
        self.num_blocks = num_blocks

        # TODO need to get input and output dimensions right.
        self.block_diag_mask = jax.scipy.linalg.block_diag(
            *[jnp.ones(block_shape, int) for _ in range(num_blocks)]
        )
        self.block_tril_mask = tril_block_mask(num_blocks, block_shape)

    def transform():
        pass

    def transform_and_log_abs_det_jacobian(self, x, *args, condition=...):
        return super().transform_and_log_abs_det_jacobian(x, *args, condition=condition)

    def inverse():
        NotImplementedError("Inverse is not implemented for this bijection.")


def tril_block_mask(num_blocks: int, block_shape: tuple):
    "Lower triangular block mask, excluding diagonal blocks."
    arrays = []
    for i in range(num_blocks):
        row = []
        for j in range(num_blocks):
            if i > j:
                row.append(jnp.ones(block_shape))
            else:
                row.append(jnp.zeros(block_shape))
        arrays.append(row)
    return jnp.block(arrays)

