import pytest
from flowjax.bijections.bnaf import BlockAutoregressiveNetwork, TanhBNAF
from flowjax.masks import block_diag_mask
import jax.numpy as jnp
from jax import random
import jax
from jax.scipy.linalg import block_diag


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

    # Check conditioning works
    barn = BlockAutoregressiveNetwork(key, dim, cond_dim, depth=1, block_dim=4)
    y1, y2 = barn.transform(x, jnp.ones(cond_dim)), barn.transform(
        x, jnp.zeros(cond_dim)
    )
    assert jnp.all(y1 != y2)


def test_TanhBNAF():
    n_blocks = 2
    block_size = 3
    x = random.uniform(random.PRNGKey(0), (n_blocks * block_size,))
    tanh = TanhBNAF(n_blocks)

    y, log_det_3d = tanh(x)
    auto_jacobian = jax.jacobian(lambda a: tanh(a)[0])(x)
    mask = block_diag_mask((block_size, block_size), n_blocks)
    assert block_diag(*jnp.exp(log_det_3d)) == pytest.approx(
        auto_jacobian * mask, abs=1e-7
    )
