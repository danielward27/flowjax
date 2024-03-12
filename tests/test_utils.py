import jax.numpy as jnp
import pytest

from flowjax.utils import _get_ufunc_signature, merge_cond_shapes

test_cases = [
    # arrays, expected_shape
    ((jnp.ones(3), jnp.ones(3)), (3,)),
    ((jnp.ones(3), 1.0), (3,)),
    ((1.0, jnp.ones(3)), (3,)),
    ((1.0, 1.0), (1,)),
    (((1.0),), (1,)),
]

test_cases = [
    # input, expected
    ([(2, 3), (2, 3), (2, 3)], (2, 3)),
    ([(), ()], ()),
    ([None, (2, 3), (2, 3)], (2, 3)),
    ([(7,), (7,)], (7,)),
]


@pytest.mark.parametrize(("input_", "expected"), test_cases)
def test_merge_shapes(input_, expected):
    assert merge_cond_shapes(input_) == expected


test_cases_error = [[(2, 3), (2, 1)], [(2, 3), (4, 2, 3)]]


@pytest.mark.parametrize("input_", test_cases_error)
def test_merge_shapes_errors(input_):
    with pytest.raises(ValueError, match="The shapes do not match."):
        merge_cond_shapes(input_)


test_cases = [
    [([(1, 2)], [(3, 4)]), "(1,2)->(3,4)"],
    [([(1, 2), (3, 4)], [(5, 6)]), "(1,2),(3,4)->(5,6)"],
    [([(1, 2)], [(3, 4), (5, 6)]), "(1,2)->(3,4),(5,6)"],
    [([(5,)], [(2,)]), "(5)->(2)"],
    [([()], [()]), "()->()"],
]


@pytest.mark.parametrize(("input_", "expected"), test_cases)
def test_get_ufunc_signature(input_, expected):
    assert _get_ufunc_signature(*input_) == expected
