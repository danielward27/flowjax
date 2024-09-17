import equinox as eqx
import jax.numpy as jnp
from jax import random

from flowjax.bijections import Affine
from flowjax.distributions import Normal, Transformed
from flowjax.train.data_fit import fit_to_data


def test_data_fit():
    dim = 3
    mean, std = jnp.ones(dim), jnp.ones(dim)
    base_dist = Normal(mean, std)
    flow = Transformed(base_dist, Affine(jnp.ones(dim), jnp.ones(dim)))

    # All params should change by default
    before = eqx.filter(flow, eqx.is_inexact_array)
    x = random.normal(random.key(0), (100, dim))
    flow, _ = fit_to_data(
        random.key(0),
        dist=flow,
        x=x,
        max_epochs=1,
        batch_size=50,
    )
    after = eqx.filter(flow, eqx.is_inexact_array)

    assert jnp.all(before.base_dist.bijection.loc != after.base_dist.bijection.loc)
    assert jnp.all(before.bijection.loc != after.bijection.loc)
