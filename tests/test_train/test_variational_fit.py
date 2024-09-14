import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from flowjax.distributions import Normal, StandardNormal
from flowjax.train.losses import ElboLoss
from flowjax.train.variational_fit import fit_to_variational_target

test_shapes = [(), (2,), (2, 3, 4)]


@pytest.mark.parametrize("shape", test_shapes)
def test_elbo_loss(shape):
    "Check finite scaler loss."
    target = StandardNormal(shape)
    vi_dist = StandardNormal(shape)
    loss = ElboLoss(target.log_prob, num_samples=100)
    loss_val = loss(*eqx.partition(vi_dist, eqx.is_inexact_array), jr.key(0))
    assert loss_val.shape == ()  # expect scalar loss
    assert jnp.isfinite(loss_val)  # expect finite loss


@pytest.mark.parametrize("shape", test_shapes)
def test_fit_to_variational_target(shape):
    "Check that loss decreases."
    vi_dist = Normal(jnp.ones(shape))
    target_dist = StandardNormal(shape)

    loss = ElboLoss(target_dist.log_prob, 50)

    vi_dist, losses = fit_to_variational_target(
        key=jr.key(0),
        dist=vi_dist,
        loss_fn=loss,
        show_progress=False,
        learning_rate=0.1,
    )
    # We expect the loss to be decreasing
    start, end = jnp.split(jnp.array(losses), 2)
    assert jnp.mean(start) > jnp.mean(end)
