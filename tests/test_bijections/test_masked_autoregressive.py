import jax.numpy as jnp
from jax import random

from flowjax.bijections.masked_autoregressive import masked_autoregressive_mlp
from flowjax.masks import rank_based_mask
from flowjax.wrappers import unwrap


def test_masked_autoregressive_mlp():
    key = random.key(0)
    in_size = 4
    in_ranks = jnp.arange(in_size)
    hidden_ranks = jnp.arange(6) % in_size
    out_ranks = jnp.arange(in_size).repeat(2)

    # Extract masks before unwrapping
    mlp = masked_autoregressive_mlp(in_ranks, hidden_ranks, out_ranks, depth=3, key=key)
    mlp = unwrap(mlp)
    masks = [layer.weight != 0 for layer in mlp.layers]
    x = jnp.ones(in_size)
    y = mlp(x)
    assert y.shape == out_ranks.shape

    # Check connectivity matrix matches expected dependency structure
    a = masks[-1].astype(int)
    for mask in reversed(masks[:-1]):
        a = a @ mask.astype(int)

    expected = rank_based_mask(in_ranks, out_ranks)
    assert jnp.all((a > 0) == expected)
