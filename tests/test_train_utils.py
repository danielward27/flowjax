from jaxflows.train_utils import count_fruitless
import pytest
from jaxflows.train_utils import train_val_split, random_permutation_multiple
import jax.numpy as jnp
from jax import random


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


def test_random_permutation_multiple():
    x = jnp.arange(10).reshape((5, 2))
    y = jnp.arange(15).reshape((5, 3))
    before = jnp.sort(jnp.concatenate((x, y), axis=1).sum(axis=1))
    x2, y2 = random_permutation_multiple(random.PRNGKey(0), (x, y))
    after = jnp.sort(jnp.concatenate((x2, y2), axis=1).sum(axis=1))
    assert (before == after).all()

