import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import random

from flowjax.bijections import Affine
from flowjax.distributions import Normal, StandardNormal, Transformed
from flowjax.train.loops import fit_to_data, fit_to_key_based_loss
from flowjax.train.losses import ElboLoss


def test_data_fit():
    dim = 3
    mean, std = jnp.ones(dim), jnp.ones(dim)
    base_dist = Normal(mean, std)
    flow = Transformed(base_dist, Affine(jnp.ones(dim), jnp.ones(dim)))

    # All params should change by default
    before = eqx.filter(flow, eqx.is_inexact_array)
    x = random.normal(random.key(0), (100, dim))
    flow, losses = fit_to_data(
        random.key(0),
        dist=flow,
        x=x,
        max_epochs=1,
        batch_size=50,
    )
    after = eqx.filter(flow, eqx.is_inexact_array)

    assert jnp.all(before.base_dist.bijection.loc != after.base_dist.bijection.loc)
    assert jnp.all(before.bijection.loc != after.bijection.loc)
    assert isinstance(losses["train"][0], float)
    assert isinstance(losses["val"][0], float)


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
