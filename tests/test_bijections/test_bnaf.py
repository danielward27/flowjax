import jax
import jax.numpy as jnp
import pytest
from jax import random

from flowjax.bijections.block_autoregressive_network import (
    BlockAutoregressiveNetwork,
    block_autoregressive_linear,
)
from flowjax.wrappers import unwrap


def test_block_autoregressive_linear():
    block_shape = (3, 2)
    linear, log_jac_3d_fn = block_autoregressive_linear(
        jax.random.key(0),
        n_blocks=3,
        block_shape=block_shape,
    )
    linear = unwrap(linear)  # Applies masking
    log_jac_3d = log_jac_3d_fn(linear)
    assert log_jac_3d.shape == (3, *block_shape)
    assert jnp.all(jnp.isfinite(log_jac_3d))


def test_BlockAutoregressiveNetwork():
    dim = 3
    x = jnp.ones(dim)
    key = random.key(0)

    barn = BlockAutoregressiveNetwork(key, dim=dim, cond_dim=None, depth=1, block_dim=4)
    barn = unwrap(barn)
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
    key = random.key(0)
    barn = BlockAutoregressiveNetwork(
        key,
        dim=dim,
        cond_dim=cond_dim,
        depth=1,
        block_dim=4,
    )
    y1 = barn.transform(x, jnp.ones(cond_dim))
    y2 = barn.transform(x, jnp.zeros(cond_dim))
    assert jnp.all(y1 != y2)
