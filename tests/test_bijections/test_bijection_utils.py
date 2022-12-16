from flowjax.bijections import Partial
from flowjax.bijections import Affine
import pytest
import jax.numpy as jnp
from flowjax.bijections import Permute
from jax.experimental.checkify import JaxRuntimeError

test_cases = {
    # name: idx, num_transformed, expected
    "int": (1, 1, jnp.array([False, True, False, False])),
    "bool_array": (
        jnp.array([True, False, True, False]),
        2,
        jnp.array([True, False, True, False]),
    ),
    "int_arry": (jnp.array([0, 2]), 2, jnp.array([True, False, True, False])),
}


@pytest.mark.parametrize(
    "idx,num_transformed,expected", test_cases.values(), ids=test_cases.keys()
)
def test_partial(idx, num_transformed, expected):
    "Check values only change where we expect"
    x = jnp.zeros(4)
    bijection = Partial(Affine(jnp.ones(num_transformed)), idx)
    y = bijection.transform(x)
    assert jnp.all((x != y) == expected)

def test_Permute_argcheck():
    with pytest.raises(JaxRuntimeError):
        Permute(jnp.array([0,0]))
