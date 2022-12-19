import jax
import jax.numpy as jnp
import pytest
from jax import random
from jax.scipy.linalg import block_diag

from flowjax.bijections.block_autoregressive_network import BlockAutoregressiveNetwork, BlockTanh
from flowjax.masks import block_diag_mask


def test_BlockAutoregressiveNetwork():
    dim = 3
    cond_dim = 2
    x = jnp.ones(dim)
    key = random.PRNGKey(0)

    barn = BlockAutoregressiveNetwork(key, dim, 0, depth=1, block_dim=4)
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
    barn = BlockAutoregressiveNetwork(key, dim, cond_dim, depth=1, block_dim=4)
    y1 = barn.transform(x, jnp.ones(cond_dim))
    y2 = barn.transform(x, jnp.zeros(cond_dim))
    assert jnp.all(y1 != y2)


def test_BlockTanh():
    n_blocks = 2
    block_size = 3
    x = random.uniform(random.PRNGKey(0), (n_blocks * block_size,))
    tanh = BlockTanh(n_blocks)

    y, log_det_3d = tanh(x)
    auto_jacobian = jax.jacobian(lambda a: tanh(a)[0])(x)
    mask = block_diag_mask((block_size, block_size), n_blocks)
    assert block_diag(*jnp.exp(log_det_3d)) == pytest.approx(
        auto_jacobian * mask, abs=1e-7
    )
