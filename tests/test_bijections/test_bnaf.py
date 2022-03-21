from flowjax.bijections.bnaf import BlockAutoregressiveLinear, block_tril_mask
import jax.numpy as jnp
import pytest


def test_block_tril_mask():
    block_shapes = [(2, 1), (2, 2), (1, 2)]
    expected = jnp.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
        ]
    )

    result = block_tril_mask(block_shapes)
    assert jnp.all(expected == result)


def test_BlockMaskedDense():
    # Test diag mask formed correctly
    layer = BlockAutoregressiveLinear(3, (2, 2))
    expected = jnp.array(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
        ]
    )
    assert jnp.all(layer.block_diag_mask == expected)

