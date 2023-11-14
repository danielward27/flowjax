import jax
import jax.numpy as jnp
import pytest
from jax import random

from flowjax.bijections.block_autoregressive_network import BlockAutoregressiveNetwork


def test_BlockAutoregressiveNetwork():
    dim = 3
    x = jnp.ones(dim)
    key = random.PRNGKey(0)

    barn = BlockAutoregressiveNetwork(key, dim=dim, cond_dim=None, depth=1, block_dim=4)
    y = barn.transform(x)
    assert y.shape == (dim,)
    auto_jacobian = jax.jacobian(barn.transform)(x)

    # Check autograd autoregressive
    assert jnp.all(jnp.triu(auto_jacobian, 1) == pytest.approx(0, abs=1e-7))
    assert jnp.all(jnp.diag(auto_jacobian) > 0)


def test_BlockAutoregressiveNetwork_conditioning():
    dim = 3
    cond_dim = 2
    x = jnp.ones(dim)
    key = random.PRNGKey(0)
    barn = BlockAutoregressiveNetwork(
        key, dim=dim, cond_dim=cond_dim, depth=1, block_dim=4,
    )
    y1 = barn.transform(x, jnp.ones(cond_dim))
    y2 = barn.transform(x, jnp.zeros(cond_dim))
    assert jnp.all(y1 != y2)
