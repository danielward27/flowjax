import jax.numpy as jnp
import pytest
from equinox import EquinoxRuntimeError

from flowjax.bijections import Affine, Partial, Permute

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
    bijection = Partial(Affine(jnp.ones(shape)), idx, x.shape)
    y = bijection.transform(x)
    assert jnp.all((x != y) == expected)


def test_Permute_argcheck():
    with pytest.raises(EquinoxRuntimeError):
        Permute(jnp.array([0, 0]))
