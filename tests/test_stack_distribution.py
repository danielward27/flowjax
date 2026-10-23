import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.scipy.stats import expon, norm

from flowjax.bijections import AdditiveCondition
from flowjax.distributions import Exponential, Normal, Stack, Transformed
from flowjax.flows import masked_autoregressive_flow


@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
def test_stack_axes(shape):
    components = [Normal(jnp.zeros(shape)), Exponential(jnp.ones(shape))]
    key = jr.key(0)
    for axis in range(-len(shape) - 1, len(shape) + 1):
        dist = Stack(components, axis=axis)
        expected_shape = np.stack([np.empty(shape)] * 2, axis=axis).shape
        assert dist.shape == expected_shape
        samples, log_probs = dist.sample_and_log_prob(key, (4,))
        assert samples.shape == (4, *expected_shape)
        assert log_probs.shape == (4,)
        np.testing.assert_allclose(dist.sample(key, (4,)), samples)
        np.testing.assert_allclose(dist.log_prob(samples), log_probs, rtol=1e-5)


def test_stack_density_and_gradients():
    def log_prob(loc, x):
        return Stack([Normal(loc, 2), Exponential(3)]).log_prob(x)

    x = jnp.array([-1.0, 2.0])
    reference = norm.logpdf(x[0], 0.5, 2) + expon.logpdf(x[1], scale=1 / 3)
    np.testing.assert_allclose(jax.jit(log_prob)(0.5, x), reference)
    np.testing.assert_allclose(jax.grad(log_prob)(0.5, x), (x[0] - 0.5) / 4)
    np.testing.assert_allclose(
        jax.grad(log_prob, argnums=1)(0.5, x),
        jnp.array([0.375, -3]),
    )
    assert jnp.isneginf(log_prob(0.5, jnp.array([0.0, -1.0])))


def test_stack_independent_samples():
    # Reusing the same key for components would produce identical coordinates.
    dist = Stack([Normal(), Normal()])
    samples = eqx.filter_jit(dist.sample)(jr.key(1), (4000,))
    assert abs(float(jnp.corrcoef(samples.T)[0, 1])) < 0.05
    np.testing.assert_allclose(jnp.mean(samples, axis=0), 0, atol=0.05)
    np.testing.assert_allclose(jnp.std(samples, axis=0), 1, atol=0.05)


def test_stack_mixed_conditions():
    conditional = Transformed(
        Normal(jnp.zeros(2)),
        AdditiveCondition(lambda c: c, (2,), (2,)),
    )
    dist = Stack([conditional, Normal(jnp.ones(2))], axis=-1)
    assert dist.cond_shape == (2,)
    condition = jnp.arange(6.0).reshape(3, 2)
    x, log_probs = eqx.filter_jit(dist.sample_and_log_prob)(jr.key(0), (4,), condition)
    assert x.shape == (4, 3, 2, 2)
    assert log_probs.shape == (4, 3)
    reference = conditional.log_prob(x[..., 0], condition)
    reference += Normal(jnp.ones(2)).log_prob(x[..., 1])
    np.testing.assert_allclose(dist.log_prob(x, condition), reference, rtol=1e-5)
    np.testing.assert_allclose(log_probs, reference, rtol=1e-5)


def test_stack_flow_base():
    flow = masked_autoregressive_flow(
        jr.key(0),
        base_dist=Stack([Normal(), Exponential(1)]),
        flow_layers=2,
        nn_width=4,
    )
    samples, log_probs = eqx.filter_jit(flow.sample_and_log_prob)(jr.key(1), (5,))
    assert samples.shape == (5, 2)
    assert jnp.all(jnp.isfinite(log_probs))
    np.testing.assert_allclose(flow.log_prob(samples), log_probs, rtol=1e-4)


def test_stack_validation():
    with pytest.raises(ValueError, match="at least one"):
        Stack([])
    with pytest.raises(ValueError, match="Expected shapes to match"):
        Stack([Normal(), Normal(jnp.zeros(2))])
    for axis in (-2, 1):
        with pytest.raises(ValueError, match="out of bounds"):
            Stack([Normal()], axis=axis)
    cond1 = Transformed(Normal(), AdditiveCondition(lambda c: c.sum(), (), (1,)))
    cond2 = Transformed(Normal(), AdditiveCondition(lambda c: c.sum(), (), (2,)))
    with pytest.raises(ValueError, match="shapes do not match"):
        Stack([cond1, cond2])
