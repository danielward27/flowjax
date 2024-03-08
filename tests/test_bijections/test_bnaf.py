import jax
import jax.numpy as jnp
import pytest
from jax import random
from jax.scipy.linalg import block_diag

from flowjax.bijections.block_autoregressive_network import (
    BlockAutoregressiveLinear,
    BlockAutoregressiveNetwork,
)
from flowjax.wrappers import unwrap


def test_BlockAutoregressiveLinear():
    block_shape = (3, 2)
    layer = BlockAutoregressiveLinear(
        jax.random.PRNGKey(0),
        n_blocks=3,
        block_shape=block_shape,
    )
    block_diag_mask = layer.linear.weight.weight.cond
    layer = unwrap(layer)  # Applies masking
    x = jnp.ones(6)
    a, log_jac_3d = layer(x)
    assert log_jac_3d.shape == (3, *block_shape)

    # Check block diag log jacobian matches autodif.
    auto_jacobian = jax.jacobian(lambda x: layer(x)[0])(x) * block_diag_mask
    assert block_diag(*jnp.exp(log_jac_3d)) == pytest.approx(auto_jacobian, abs=1e-7)


def test_BlockAutoregressiveNetwork():
    dim = 3
    x = jnp.ones(dim)
    key = random.PRNGKey(0)

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
    key = random.PRNGKey(0)
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
