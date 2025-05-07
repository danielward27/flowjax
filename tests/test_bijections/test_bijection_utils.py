from functools import partial

import jax
import jax.numpy as jnp
import pytest
from equinox import EquinoxRuntimeError

from flowjax.bijections import Affine, Indexed, NumericalInverse, Permute
from flowjax.root_finding import bisection_search, root_finder_to_inverter

test_cases = {
    # name: idx, expected
    "int": (1, jnp.array([False, True, False, False])),
    "bool_array": (
        jnp.array([True, False, True, False]),
        jnp.array([True, False, True, False]),
    ),
    "int_arry": (jnp.array([0, 2]), jnp.array([True, False, True, False])),
}


@pytest.mark.parametrize(
    ("idx", "expected"),
    test_cases.values(),
    ids=test_cases.keys(),
)
def test_partial(idx, expected):
    "Check values only change where we expect."
    x = jnp.zeros(4)
    shape = x[idx].shape
    bijection = Indexed(Affine(jnp.ones(shape)), idx, x.shape)
    y = bijection.transform(x)
    assert jnp.all((x != y) == expected)


def test_Permute_argcheck():
    with pytest.raises(EquinoxRuntimeError):
        Permute(jnp.array([0, 0]))


test_cases = [jax.grad, jax.jacfwd, jax.jacrev]


@pytest.mark.parametrize("diff_fn", test_cases)
def test_NumericalInverse_not_differentiable(diff_fn):
    bijection = NumericalInverse(
        Affine(5, 2),
        root_finder_to_inverter(
            partial(bisection_search, lower=-1, upper=1, atol=1e-7),
        ),
    )
    with pytest.raises(
        RuntimeError, match="Computing gradients through the numerical inverse"
    ):
        diff_fn(bijection.inverse)(jnp.ones(()))
