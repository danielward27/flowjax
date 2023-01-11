import jax.numpy as jnp
import pytest
import jax.random as jr

from flowjax.utils import tile_until_length, inv_cum_sum, broadcast_shapes, check_shapes, _get_ufunc_signature


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


@pytest.mark.parametrize("key", jr.split(jr.PRNGKey(0), 5))
def test_inv_cum_sum(key):
    x = jr.uniform(key, (10,))
    x_cumsum = jnp.cumsum(x)
    x_recon = inv_cum_sum(x_cumsum)
    assert pytest.approx(x, abs=1e-6) == x_recon



test_cases = [
    # input, expected
    ([(2, 3), (2,3), (2,3)], (2,3)),
    ([(), ()], ()),
    ([None, (2,3), (2,3)], (2,3)),
    ([(7,), (7,)], (7,)),
]

@pytest.mark.parametrize("input", [t[0] for t in test_cases])
def test_check_shapes(input):
    check_shapes(input)

@pytest.mark.parametrize("input,expected", test_cases)
def test_broadcast_shapes(input, expected):
    assert broadcast_shapes(input) == expected



test_cases_error = [
    [(2,3), (2,1)],
    [(2, 3), (4, 2, 3)]
]

@pytest.mark.parametrize("input", test_cases_error)
def test_check_shapes_errors(input):
    with pytest.raises(ValueError):
        check_shapes(input)

@pytest.mark.parametrize("input", test_cases_error)
def test_broadcast_shapes_errors(input):
    with pytest.raises(ValueError):
        broadcast_shapes(input)


test_cases = [
    [([(1,2)], [(3,4)]), "(1,2)->(3,4)"],
    [([(1,2),(3,4)], [(5,6)]), "(1,2),(3,4)->(5,6)"],
    [([(1,2)], [(3,4), (5,6)]), "(1,2)->(3,4),(5,6)"    ],
    [([(5,)], [(2,)]), "(5)->(2)"],
    [([()], [()]), "()->()"],
]

@pytest.mark.parametrize("input,expected", test_cases)
def test_get_ufunc_signature(input, expected):
    assert _get_ufunc_signature(*input) == expected
