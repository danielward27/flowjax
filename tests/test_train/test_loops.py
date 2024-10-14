import jax.numpy as jnp
import jax.random as jr
import pytest

from flowjax.distributions import Normal, StandardNormal
from flowjax.train.loops import fit_to_key_based_loss
from flowjax.train.losses import ElboLoss

test_shapes = [(), (2,), (2, 3, 4)]


@pytest.mark.parametrize("shape", test_shapes)
def test_fit_to_key_based_loss(shape):
    "Check that loss decreases."
    vi_dist = Normal(jnp.ones(shape))
    target_dist = StandardNormal(shape)

    loss = ElboLoss(target_dist.log_prob, 50)

    vi_dist, losses = fit_to_key_based_loss(
        key=jr.key(0),
        tree=vi_dist,
        loss_fn=loss,
        show_progress=False,
        steps=50,
        learning_rate=1e-2,
    )
    # We expect the loss to be decreasing
    start, end = jnp.split(jnp.array(losses), 2)
    assert jnp.mean(start) > jnp.mean(end)
    assert isinstance(losses[0], float)
