"Tests for train_utils.py"
import jax.numpy as jnp
import jax.random as jr
import pytest

from flowjax.train.train_utils import train_val_split


def test_train_val_split():
    key = jr.PRNGKey(0)
    arrays = [jnp.arange(10).reshape(5, 2) for _ in range(3)]

    train, val = train_val_split(key, arrays, val_prop=0.2)

    # Check shapes
    assert all(train_arr.shape == (4, 2) for train_arr in train)
    assert all(val_arr.shape == (1, 2) for val_arr in val)

    arrays = [jnp.ones((5, 2)), jnp.ones((3, 5))]

    # Axis mismatch
    with pytest.raises(ValueError):
        train_val_split(key, arrays, val_prop=0.2)

    # Axes as list
    train, val = train_val_split(key, arrays, val_prop=0.2, axis=[0, 1])
    assert train[0].shape == (4, 2)
    assert val[0].shape == (1, 2)
    assert train[1].shape == (3, 4)
    assert val[1].shape == (3, 1)
