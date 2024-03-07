import jax
import jax.numpy as jnp
import pytest
from jax import random
from jax.scipy.linalg import block_diag

from flowjax.masks import rank_based_mask
from flowjax.nn import BlockAutoregressiveLinear, autoregressive_mlp
from flowjax.wrappers import unwrap


def test_BlockAutoregressiveLinear():
    block_shape = (3, 2)
    layer = BlockAutoregressiveLinear(
        jax.random.PRNGKey(0),
        n_blocks=3,
        block_shape=block_shape,
    )
    x = jnp.ones(6)
    a, log_jac_3d = unwrap(layer)(x)
    assert log_jac_3d.shape == (3, *block_shape)

    # Check block diag log jacobian matches autodif.
    auto_jacobian = jax.jacobian(lambda x: layer(x)[0])(x) * layer.b_diag_mask
    assert block_diag(*jnp.exp(log_jac_3d)) == pytest.approx(auto_jacobian, abs=1e-7)


def test_autoregressive_mlp():
    key = random.PRNGKey(0)
    in_size = 4
    in_ranks = jnp.arange(in_size)
    hidden_ranks = jnp.arange(6) % in_size
    out_ranks = jnp.arange(in_size).repeat(2)

    # Extract masks before unwrapping
    mlp = autoregressive_mlp(in_ranks, hidden_ranks, out_ranks, depth=3, key=key)
    masks = [layer.cond for layer in mlp.layers]

    mlp = unwrap(mlp)
    x = jnp.ones(in_size)
    y = mlp(x)
    assert y.shape == out_ranks.shape

    # Check connectivity matrix matches expected dependency structure
    a = masks[-1].mask
    for mask in reversed(masks[:-1]):
        a = a @ mask

    expected = rank_based_mask(jnp.arange(in_size), out_ranks)
    assert jnp.all((a > 0) == expected)
