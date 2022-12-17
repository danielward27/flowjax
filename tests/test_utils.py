import jax.numpy as jnp
import pytest

from flowjax.utils import broadcast_arrays_1d, tile_until_length


def test_tile_until_length():
    x = jnp.array([1, 2])

    y = tile_until_length(x, 4)
    assert jnp.all(y == jnp.array([1, 2, 1, 2]))

    y = tile_until_length(x, 3)
    assert jnp.all(y == jnp.array([1, 2, 1]))

    y = tile_until_length(x, 1)
    assert jnp.all(y == jnp.array([1]))


test_cases = [
    # arrays, expected_shape
    ((jnp.ones(3), jnp.ones(3)), (3,)),
    ((jnp.ones(3), 1.0), (3,)),
    ((1.0, jnp.ones(3)), (3,)),
    ((1.0, 1.0), (1,)),
    (((1.0),), (1,)),
]


@pytest.mark.parametrize("arrays,expected", test_cases)
def test_broadcast_arrays_1d(arrays, expected):
    out = broadcast_arrays_1d(*arrays)
    assert len(arrays) == len(out)
    match_expected = [a.shape == expected for a in out]
    assert jnp.all(jnp.array(match_expected))
