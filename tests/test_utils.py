import pytest
from flowjax.utils import tile_until_length
import jax.numpy as jnp


def test_tile_until_length():
    x = jnp.array([1, 2])

    y = tile_until_length(x, 4)
    assert jnp.all(y == jnp.array([1, 2, 1, 2]))

    y = tile_until_length(x, 3)
    assert jnp.all(y == jnp.array([1, 2, 1]))

    y = tile_until_length(x, 1)
    assert jnp.all(y == jnp.array([1]))
