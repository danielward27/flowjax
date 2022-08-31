import pytest
import jax.numpy as jnp
import jax
from jax import random
import equinox as eqx
from flowjax.bijections.masked_autoregressive import (
    AutoregressiveMLP,
    MaskedLinear,
    rank_based_mask,
    # rank_based_mask_expand,
)


def test_rank_based_mask():
    in_ranks = jnp.arange(2)
    out_ranks = jnp.array([0, 1, 1, 2])

    expected_mask = jnp.array([[0, 0], [1, 0], [1, 0], [1, 1]], dtype=jnp.int32)

    mask = rank_based_mask(in_ranks, out_ranks)
    assert jnp.all(expected_mask == mask)

    in_ranks = jnp.array([0, 0, 1, 1])
    out_ranks = jnp.array([0, 1])

    expected_mask = jnp.array([[0, 0, 0, 0], [1, 1, 0, 0]], dtype=jnp.int32)
    mask = rank_based_mask(in_ranks, out_ranks)
    assert jnp.all(expected_mask == mask)


def test_MaskedLinear():
    key = random.PRNGKey(0)
    in_ranks = jnp.arange(3)
    out_ranks = jnp.arange(5).repeat(2)
    mask = rank_based_mask(in_ranks, out_ranks)
    layer = MaskedLinear(mask, key=key)
    x = jnp.ones(len(in_ranks))

    # Ensure dependency structure correctly enforced
    non_zero_jac = jnp.abs(jax.jacobian(layer)(x)) > 1e-8
    expected = rank_based_mask(in_ranks, out_ranks)
    assert jnp.all(expected == non_zero_jac)


def test_AutoregressiveMLP():
    key = random.PRNGKey(0)
    in_size = 4
    out_ranks = jnp.arange(in_size).repeat(2)
    mlp = AutoregressiveMLP(in_size, out_ranks, width_size=6, depth=3, key=key)
    x = jnp.ones(in_size)
    y = mlp(x)
    assert y.shape == out_ranks.shape

    # Check connectivity matrix matches expected dependency structure
    a = mlp.layers[-1].mask
    for layer in reversed(mlp.layers[:-1]):
        a = a @ layer.mask

    expected = rank_based_mask(jnp.arange(in_size), out_ranks)
    assert jnp.all((a > 0) == expected)
