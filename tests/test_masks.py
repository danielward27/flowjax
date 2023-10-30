import jax.numpy as jnp

from flowjax.masks import block_diag_mask, block_tril_mask, rank_based_mask


def test_rank_based_mask():
    in_ranks = jnp.arange(2)
    out_ranks = jnp.array([0, 1, 1, 2])

    expected_mask = jnp.array([[0, 0], [1, 0], [1, 0], [1, 1]], dtype=jnp.int32)

    mask = rank_based_mask(in_ranks, out_ranks)
    assert mask.dtype == jnp.int32
    assert jnp.all(expected_mask == mask)

    in_ranks = jnp.array([0, 0, 1, 1])
    out_ranks = jnp.array([0, 1])

    expected_mask = jnp.array([[0, 0, 0, 0], [1, 1, 0, 0]], dtype=jnp.int32)
    mask = rank_based_mask(in_ranks, out_ranks)
    assert mask.dtype == jnp.int32
    assert jnp.all(expected_mask == mask)


def test_block_tril_mask():
    args = [(1, 2), 3]
    expected = jnp.array([[0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    mask = block_tril_mask(*args)
    assert mask.dtype == jnp.int32
    assert jnp.all(expected == mask)


def test_block_diag_mask():
    args = [(1, 2), 3]
    expected = jnp.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]])
    mask = block_diag_mask(*args)
    assert mask.dtype == jnp.int32
    assert jnp.all(expected == mask)
