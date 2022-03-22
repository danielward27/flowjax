#%%
from flowjax.bijections.bnaf import (
    BlockAutoregressiveLinear,
    _IdentityBNAF,
    b_diag_mask,
    b_tril_mask,
    BlockAutoregressiveNetwork,
)
import jax.numpy as jnp
from jax import random
import jax
import pytest
from flowjax.bijections.bnaf import _TanhBNAF
from jax.scipy.linalg import block_diag


# %%
def test_b_tril_mask():
    args = [(1, 2), 3]
    expected = jnp.array([[0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    result = b_tril_mask(*args)
    assert jnp.all(expected == result)


def test_b_diag_mask():
    args = [(1, 2), 3]
    expected = jnp.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]])
    result = b_diag_mask(*args)
    assert jnp.all(expected == result)


def test_BlockAutoregressiveLinear():
    block_shape = (3, 2)
    layer = BlockAutoregressiveLinear(random.PRNGKey(0), 3, block_shape)
    x = jnp.ones(6)
    a, log_jac_3d = layer(x)
    assert log_jac_3d.shape == (3, *block_shape)

    # Check block diag log jacobian matches autodif.
    auto_jacobian = jax.jacobian(lambda x: layer(x)[0])(x) * layer.b_diag_mask
    assert block_diag(*jnp.exp(log_jac_3d)) == pytest.approx(auto_jacobian, abs=1e-7)


def test_BlockAutoregressiveNetwork():
    dim = 5
    x = jnp.ones(dim)

    for activation in [_IdentityBNAF, _TanhBNAF]:
        barn = BlockAutoregressiveNetwork(random.PRNGKey(0), dim, activation=activation)
        y = barn.transform(x)
        assert y.shape == (dim,)
        auto_jacobian = jax.jacobian(barn.transform)(x)

        # Check autograd autoregressive
        assert jnp.all(jnp.triu(auto_jacobian, 1) == pytest.approx(0, abs=1e-7))
        assert jnp.all(jnp.diag(auto_jacobian) > 0)

        # Check log abs det jacobian calculation
        y, log_abs_det_jacobian = barn.transform_and_log_abs_det_jacobian(x)
        expected = jnp.log(jnp.diag(auto_jacobian)).sum()
        assert log_abs_det_jacobian == pytest.approx(expected)


from flowjax.bijections.bnaf import b_diag_mask


def test__TanhBNAF():
    num_blocks = 2
    block_size = 3
    x = random.uniform(random.PRNGKey(0), (num_blocks * block_size,))
    tanh = _TanhBNAF(num_blocks)

    y, log_det_3d = tanh(x)
    auto_jacobian = jax.jacobian(lambda a: tanh(a)[0])(x)
    mask = b_diag_mask((block_size, block_size), num_blocks)
    # We have a bunch of ones in log_det_3d
    assert block_diag(*jnp.exp(log_det_3d)) == pytest.approx(
        auto_jacobian * mask, abs=1e-7
    )


# This is where we wnt wrong

# # %%
# import jax.numpy as jnp
# import jax
# from flowjax.bijections.bnaf import BlockAutoregressiveNetwork
# import equinox as eqx
# from jax import random

# dim = 5
# barn = BlockAutoregressiveNetwork(random.PRNGKey(0), dim)
# x = random.uniform(random.PRNGKey(0), (dim,))
# y = barn.transform(x)
# assert y.shape == (5,)

# y, logdet = barn.transform_and_log_abs_det_jacobian(x)

# logdet.reshape(x.shape)

# # %%
# # IS equation 10 a scaler?
# # Jacobian is upper triangular. But somewhere we are computing it incorrectly.
# # Maybe we should just try and train it and see if it fails.
# # 9 does not compute the full jacobian, just the diagonal elements..
# # %%
# expected_log_jac = jnp.log(jax.jacobian(barn.transform)(x))
# print(expected_log_jac)
# print(jnp.diagonal(expected_log_jac).sum())
# # %%
# # def f(x):
# #     return np.array([x[0] ** 2, x[1] ** 2])


# # x = np.array([[3.0, 11.0], [5.0, 13.0], [7.0, 17.0]])

# # jac = jax.jacobian(f)

# # %%
