import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random

from flowjax.bijections import Affine
from flowjax.distributions import Normal, Transformed
from flowjax.train.data_fit import fit_to_data


def test_data_fit_filter_spec():
    dim = 3
    mean, std = jnp.ones(dim), jnp.ones(dim)
    base_dist = Normal(mean, std)
    flow = Transformed(base_dist, Affine(jnp.ones(dim), jnp.ones(dim)))

    # All params should change by default
    before = eqx.filter(flow, eqx.is_inexact_array)
    x = random.normal(random.PRNGKey(0), (100, dim))
    flow, _ = fit_to_data(
        random.PRNGKey(0), dist=flow, x=x, max_epochs=1, batch_size=50,
    )
    after = eqx.filter(flow, eqx.is_inexact_array)

    assert jnp.all(before.base_dist.bijection.loc != after.base_dist.bijection.loc)
    assert jnp.all(before.bijection.loc != after.bijection.loc)

    # We can provide a filter spec to avoid e.g. training the base distribution.
    before = eqx.filter(flow, eqx.is_inexact_array)
    filter_spec = jtu.tree_map(eqx.is_inexact_array, flow)
    filter_spec = eqx.tree_at(lambda tree: tree.base_dist, filter_spec, replace=False)
    flow, _ = fit_to_data(
        key=random.PRNGKey(0),
        dist=flow,
        x=x,
        max_epochs=1,
        batch_size=50,
        filter_spec=filter_spec,
    )
    after = eqx.filter(flow, eqx.is_inexact_array)

    assert jnp.all(before.base_dist.bijection.loc == after.base_dist.bijection.loc)
    assert jnp.all(before.bijection.loc != after.bijection.loc)
