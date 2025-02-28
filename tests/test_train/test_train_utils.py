"Tests for train_utils.py."
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from flowjax.train.train_utils import count_fruitless, get_batches, train_val_split


def test_train_val_split():
    key = jr.key(0)
    arrays = [jnp.ones(5), jnp.ones((5, 2)), None, jnp.ones((5, 3, 3))]

    train, val = train_val_split(key, arrays, val_prop=0.2)

    train_shapes = jax.tree.map(jnp.shape, train)
    val_shapes = jax.tree.map(jnp.shape, val)

    expected_train_shapes = [(4,), (4, 2), None, (4, 3, 3)]
    expected_val_shapes = [(1,), (1, 2), None, (1, 3, 3)]

    assert all(
        train_shape == expected_train_shape
        for train_shape, expected_train_shape in zip(
            train_shapes,
            expected_train_shapes,
            strict=True,
        )
    )
    assert all(
        val_shape == expected_val_shape
        for val_shape, expected_val_shape in zip(
            val_shapes, expected_val_shapes, strict=True
        )
    )

    arrays = [jnp.ones((5, 2)), jnp.ones((3, 5)), None]

    # Axis mismatch
    with pytest.raises(ValueError, match="Array dimensions must match along axis 0"):
        train_val_split(key, arrays, val_prop=0.2)


def test_count_fruitless():
    assert count_fruitless([12.0, 2.0, 3.0, 4.0]) == 2
    assert count_fruitless([0.0]) == 0.0
    assert count_fruitless([0.0, 12.0]) == 1


def test_get_batches():
    arrays = [jnp.arange(26).reshape(13, 2), jnp.arange(26).reshape(13, 2), None]
    out = get_batches(arrays, batch_size=4)
    assert out[0].shape == (3, 4, 2)
    assert out[1].shape == (3, 4, 2)
    assert out[2] is None
