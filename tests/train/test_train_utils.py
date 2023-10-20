"Tests for train_utils.py."
import jax.numpy as jnp
import jax.random as jr
import pytest

from flowjax.train.train_utils import count_fruitless, get_batches, train_val_split


def test_train_val_split():
    key = jr.PRNGKey(0)
    arrays = [jnp.arange(10).reshape(5, 2) for _ in range(3)]

    train, val = train_val_split(key, arrays, val_prop=0.2)

    # Check shapes
    assert all(train_arr.shape == (4, 2) for train_arr in train)
    assert all(val_arr.shape == (1, 2) for val_arr in val)

    arrays = [jnp.ones((5, 2)), jnp.ones((3, 5))]

    # Axis mismatch
    with pytest.raises(ValueError, match="Array dimensions must match along axis 0"):
        train_val_split(key, arrays, val_prop=0.2)


def test_count_fruitless():
    assert count_fruitless([12, 2, 3, 4]) == 2
    assert count_fruitless([0]) == 0
    assert count_fruitless([0, 12]) == 1


def test_get_batches():
    arrays = [jnp.arange(26).reshape(13, 2)] * 2
    out = get_batches(arrays, batch_size=4)
    assert out[0].shape == (3, 4, 2)
