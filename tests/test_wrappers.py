import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from flowjax.bijections import Exp, Scale
from flowjax.distributions import Normal
from flowjax.wrappers import (
    BijectionReparam,
    Lambda,
    NonTrainable,
    WeightNormalization,
    non_trainable,
    unwrap,
)


def test_BijectionReparam():

    with pytest.raises(jax.lib.xla_extension.XlaRuntimeError, match="Exp"):
        BijectionReparam(-jnp.ones(3), Exp())

    param = jnp.array([jnp.inf, 1, 2])
    wrapped = BijectionReparam(param, Exp())
    assert pytest.approx(unwrap(wrapped)) == param
    assert pytest.approx(wrapped.arr) == jnp.log(param)

    # Test with vmapped constructor

    def _get_param(arr):
        return BijectionReparam(arr, Scale(jnp.full(3, fill_value=2)))

    init_param = jnp.ones((1, 2, 3))
    param = eqx.filter_vmap(eqx.filter_vmap(_get_param))(init_param)
    assert pytest.approx(init_param) == unwrap(param)


def test_Lambda():
    diag = Lambda(jnp.diag, jnp.ones(3))
    assert pytest.approx(jnp.eye(3)) == unwrap(diag)

    # Test works when vmapped (note diag does not follow standard vectorization rules)
    v_diag = eqx.filter_vmap(Lambda)(jnp.diag, jnp.ones((4, 3)))
    expected = eqx.filter_vmap(jnp.eye, axis_size=4)(3)
    assert pytest.approx(expected) == unwrap(v_diag)

    # Test works when double vmapped
    v_diag = eqx.filter_vmap(eqx.filter_vmap(Lambda))(jnp.diag, jnp.ones((5, 4, 3)))
    expected = eqx.filter_vmap(eqx.filter_vmap(jnp.eye, axis_size=4), axis_size=5)(3)
    assert pytest.approx(expected) == unwrap(v_diag)

    # Test works when no arrays present (in which case axis_size is relied on)
    unwrappable = eqx.filter_vmap(eqx.filter_vmap(Lambda, axis_size=2), axis_size=3)(
        lambda: jnp.zeros(())
    )
    assert pytest.approx(unwrap(unwrappable)) == jnp.zeros((3, 2))


def test_NonTrainable_and_non_trainable():
    dist1 = eqx.tree_at(lambda dist: dist.bijection, Normal(), replace_fn=NonTrainable)
    dist2 = non_trainable(Normal())

    def loss(dist, x):
        return dist.log_prob(x)

    for dist in [dist1, dist2]:
        grad = eqx.filter_grad(loss)(dist, 1)
        assert pytest.approx(0) == jax.flatten_util.ravel_pytree(grad)[0]


def test_WeightNormalization():
    arr = jr.normal(jr.PRNGKey(0), (10, 3))
    weight_norm = WeightNormalization(arr)

    # Unwrapped norms should match weightnorm scale parameter
    expected = unwrap(weight_norm.scale)
    assert pytest.approx(expected) == jnp.linalg.norm(
        unwrap(weight_norm), axis=-1, keepdims=True
    )

    # Test under vmap
    arr = jr.normal(jr.PRNGKey(0), (5, 10, 3))
    weight_norm = eqx.filter_vmap(WeightNormalization)(arr)
    expected = unwrap(weight_norm.scale)
    assert pytest.approx(expected) == eqx.filter_vmap(
        lambda arr: jnp.linalg.norm(arr, axis=1, keepdims=True)
    )(unwrap(weight_norm))
