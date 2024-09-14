"Tests for train_utils.py."
import jax.numpy as jnp
import jax.random as jr
import pytest

from flowjax.train.train_utils import count_fruitless, get_batches, train_val_split


def test_train_val_split():
    key = jr.key(0)
    arrays = [jnp.ones(5), jnp.ones((5, 2)), jnp.ones((5, 3, 3))]

    train, val = train_val_split(key, arrays, val_prop=0.2)

    expected_train = [(4,), (4, 2), (4, 3, 3)]
    expected_val = [(1,) + s[1:] for s in expected_train]

    # Check shapes
    assert all(
        train_arr.shape == expected
        for train_arr, expected in zip(train, expected_train, strict=True)
    )
    assert all(
        val_arr.shape == expected
        for val_arr, expected in zip(val, expected_val, strict=True)
    )

    arrays = [jnp.ones((5, 2)), jnp.ones((3, 5))]

    # Axis mismatch
    with pytest.raises(ValueError, match="Array dimensions must match along axis 0"):
        train_val_split(key, arrays, val_prop=0.2)


def test_count_fruitless():
    assert count_fruitless([12.0, 2.0, 3.0, 4.0]) == 2
    assert count_fruitless([0.0]) == 0.0
    assert count_fruitless([0.0, 12.0]) == 1


def test_get_batches():
    arrays = [jnp.arange(26).reshape(13, 2)] * 2
    out = get_batches(arrays, batch_size=4)
    assert out[0].shape == (3, 4, 2)
