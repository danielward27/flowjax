from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.tree_util import tree_map

from flowjax.distributions import Normal
from flowjax.wrappers import (
    NonTrainable,
    Parameterize,
    WeightNormalization,
    non_trainable,
    unwrap,
)


def test_Parameterize():
    diag = Parameterize(jnp.diag, jnp.ones(3))
    assert pytest.approx(jnp.eye(3)) == unwrap(diag)


def test_nested_Parameterized():
    param = Parameterize(
        jnp.square,
        Parameterize(jnp.square, Parameterize(jnp.square, 2)),
    )
    assert unwrap(param) == jnp.square(jnp.square(jnp.square(2)))


def test_non_trainable():
    dist = non_trainable(Normal())

    def loss(dist, x):
        return dist.log_prob(x)

    grad = eqx.filter_grad(loss)(dist, 1)
    assert pytest.approx(0) == jax.flatten_util.ravel_pytree(grad)[0]


def test_WeightNormalization():
    arr = jr.normal(jr.key(0), (10, 3))
    weight_norm = WeightNormalization(arr)

    # Unwrapped norms should match weightnorm scale parameter
    expected = unwrap(weight_norm.scale)
    assert pytest.approx(expected) == jnp.linalg.norm(
        unwrap(weight_norm), axis=-1, keepdims=True
    )


test_cases = {
    "NonTrainable": lambda key: NonTrainable(jr.normal(key, 10)),
    "Parameterize-exp": lambda key: Parameterize(jnp.exp, jr.normal(key, 10)),
    "Parameterize-diag": lambda key: Parameterize(jnp.diag, jr.normal(key, 10)),
    "WeightNormalization": lambda key: WeightNormalization(jr.normal(key, (10, 2))),
}


@pytest.mark.parametrize("shape", [(), (2,), (5, 2, 4)])
@pytest.mark.parametrize("wrapper_fn", test_cases.values(), ids=test_cases.keys())
def test_vectorization_invariance(wrapper_fn, shape):
    keys = jr.split(jr.key(0), prod(shape))
    wrapper = wrapper_fn(keys[0])  # Standard init

    # Multiple vmap init - should have same result in zero-th index
    vmap_wrapper_fn = wrapper_fn
    for _ in shape:
        vmap_wrapper_fn = eqx.filter_vmap(vmap_wrapper_fn)

    vmap_wrapper = vmap_wrapper_fn(keys.reshape(shape))

    unwrapped = unwrap(wrapper)
    unwrapped_vmap = unwrap(vmap_wrapper)
    unwrapped_vmap_zero = tree_map(
        lambda leaf: leaf[*([0] * len(shape)), ...],
        unwrapped_vmap,
    )
    assert eqx.tree_equal(unwrapped, unwrapped_vmap_zero, atol=1e-7)
