from flowjax.bijections.bnaf import BlockMaskedDense, tril_block_mask
import jax.numpy as jnp
import pytest


def test_BlockMaskedDense():
    # Test masks formed correctly
    layer = BlockMaskedDense(3, (1, 2))
    expected = jnp.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]])
    assert jnp.all(layer.block_diag_mask == expected)


def test_tril_block_mask():
    expected = jnp.array([[0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    result = tril_block_mask(3, (1, 2))
    assert jnp.all(expected == result)
