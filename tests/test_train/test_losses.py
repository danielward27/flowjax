import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from flowjax.distributions import Normal
from flowjax.train.losses import ElboLoss, _get_contrastive_idxs


def test_get_contrastive_idxs():
    key = jr.key(0)
    batch_size = 5

    for _ in range(5):
        key, subkey = jr.split(key)
        idxs = _get_contrastive_idxs(subkey, batch_size=batch_size, n_contrastive=4)
        for i, row in enumerate(idxs):
            assert i not in row

        assert jnp.all(idxs < batch_size)


test_shapes = [(), (2,), (2, 3, 4)]


@pytest.mark.parametrize("shape", test_shapes)
def test_elbo_loss(shape):
    "Check finite scaler loss."
    target = Normal(jnp.ones(shape))
    vi_dist = Normal(jnp.ones(shape))
    loss = ElboLoss(target.log_prob, num_samples=100)
    loss_val = loss(*eqx.partition(vi_dist, eqx.is_inexact_array), jr.key(0))
    assert loss_val.shape == ()  # expect scalar loss
    assert jnp.isfinite(loss_val)  # expect finite loss
