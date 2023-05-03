import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from jax import random

from flowjax.bijections import Affine
from flowjax.distributions import Normal, Transformed
from flowjax.train.data_fit import (count_fruitless, fit_to_data,
                                    train_val_split)


def test_count_fruitless():
    assert count_fruitless([12, 2, 3, 4]) == 2
    assert count_fruitless([0]) == 0
    assert count_fruitless([0, 12]) == 1


def test_train_val_split():
    x = jnp.ones((100, 10))
    y = jnp.zeros((100, 5))
    key = random.PRNGKey(0)
    train, test = train_val_split(key, (x, y))
    assert train[0].shape == (90, 10)
    assert train[1].shape == (90, 5)
    assert test[0].shape == (10, 10)
    assert test[1].shape == (10, 5)


def test_data_fit_filter_spec():
    dim = 3
    mean, std = jnp.ones(dim), jnp.ones(dim)
    base_dist = Normal(mean, std)
    flow = Transformed(base_dist, Affine(jnp.ones(dim), jnp.ones(dim)))

    # All params should change by default
    before = eqx.filter(flow, eqx.is_inexact_array)
    x = random.normal(random.PRNGKey(0), (100, dim))
    flow, _ = fit_to_data(random.PRNGKey(0), flow, x, max_epochs=1, batch_size=50)
    after = eqx.filter(flow, eqx.is_inexact_array)

    assert jnp.all(before.base_dist.bijection.loc != after.base_dist.bijection.loc)
    assert jnp.all(before.bijection.loc != after.bijection.loc)

    # But we can provide a filter spec to avoid e.g. training the base distribution parameters.
    before = eqx.filter(flow, eqx.is_inexact_array)
    filter_spec = jtu.tree_map(lambda x: eqx.is_inexact_array(x), flow)
    filter_spec = eqx.tree_at(lambda tree: tree.base_dist, filter_spec, replace=False)
    flow, _ = fit_to_data(
        random.PRNGKey(0), flow, x, max_epochs=1, batch_size=50, filter_spec=filter_spec
    )
    after = eqx.filter(flow, eqx.is_inexact_array)

    assert jnp.all(before.base_dist.bijection.loc == after.base_dist.bijection.loc)
    assert jnp.all(before.bijection.loc != after.bijection.loc)
