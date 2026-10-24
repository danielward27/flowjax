import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.scipy.stats import expon, norm

from flowjax.bijections import AdditiveCondition, Affine, NumericalInverse
from flowjax.distributions import Concatenate, Exponential, Normal, Transformed
from flowjax.flows import masked_autoregressive_flow


@pytest.mark.parametrize(
    ("shapes", "axis"),
    [
        ([(1,), (2,)], 0),
        ([(1,), (2,)], -1),
        ([(1, 3), (2, 3)], 0),
        ([(2, 1), (2, 3)], 1),
        ([(2, 1), (2, 3)], -1),
        ([(1, 3), (2, 3)], -2),
    ],
)
def test_concatenate_axes(shapes, axis):
    dist = Concatenate(
        [Normal(jnp.zeros(shapes[0])), Exponential(2 * jnp.ones(shapes[1]))], axis
    )
    expected_shape = np.concatenate([np.empty(s) for s in shapes], axis=axis).shape
    assert dist.shape == expected_shape
    samples, log_probs = eqx.filter_jit(dist.sample_and_log_prob)(jr.key(0), (4,))
    assert samples.shape == (4, *expected_shape)
    event_axis = axis % len(shapes[0])
    normal_part = jnp.take(
        samples, jnp.arange(shapes[0][event_axis]), axis=event_axis + 1
    )
    exp_part = jnp.take(
        samples,
        jnp.arange(shapes[0][event_axis], expected_shape[event_axis]),
        axis=event_axis + 1,
    )
    reference = norm.logpdf(normal_part).reshape(4, -1).sum(-1)
    reference += expon.logpdf(exp_part, scale=0.5).reshape(4, -1).sum(-1)
    np.testing.assert_allclose(log_probs, reference, rtol=1e-5)
    np.testing.assert_allclose(dist.log_prob(samples), reference, rtol=1e-5)
    np.testing.assert_allclose(dist.sample(jr.key(0), (4,)), samples)


def test_concatenate_gradients():
    def log_prob(loc, x):
        return Concatenate(
            [Normal(jnp.full(2, loc), 2), Exponential(jnp.full(1, 3))]
        ).log_prob(x)

    x = jnp.array([0.3, -0.2, 1.2])
    reference = norm.logpdf(x[:2], 0.5, 2).sum() + expon.logpdf(x[2], scale=1 / 3)
    np.testing.assert_allclose(jax.jit(log_prob)(0.5, x), reference, rtol=1e-5)
    np.testing.assert_allclose(jax.grad(log_prob)(0.5, x), ((x[:2] - 0.5) / 4).sum())
    expected = jnp.concatenate([-(x[:2] - 0.5) / 4, jnp.array([-3.0])])
    np.testing.assert_allclose(jax.grad(log_prob, argnums=1)(0.5, x), expected)


def test_concatenate_mixed_conditions():
    conditional = Transformed(
        Normal(jnp.zeros(1)), AdditiveCondition(lambda c: c[:1], (1,), (2,))
    )
    dist = Concatenate([conditional, Normal(jnp.ones(2))])
    condition = jnp.arange(6.0).reshape(3, 2)
    samples, log_probs = dist.sample_and_log_prob(jr.key(0), (4,), condition)
    assert samples.shape == (4, 3, 3)
    reference = conditional.log_prob(samples[..., :1], condition)
    reference += Normal(jnp.ones(2)).log_prob(samples[..., 1:])
    np.testing.assert_allclose(log_probs, reference, rtol=1e-5)
    np.testing.assert_allclose(dist.log_prob(samples, condition), reference, rtol=1e-5)


def test_concatenate_flow_base():
    base = Concatenate([Normal(jnp.zeros(2)), Exponential(jnp.ones(1))])
    flow = masked_autoregressive_flow(
        jr.key(0), base_dist=base, flow_layers=2, nn_width=4
    )
    samples, log_probs = eqx.filter_jit(flow.sample_and_log_prob)(jr.key(1), (5,))
    assert samples.shape == (5, 3)
    assert jnp.all(jnp.isfinite(log_probs))
    np.testing.assert_allclose(flow.log_prob(samples), log_probs, rtol=1e-4)


def test_concatenate_sampling_preserves_forward_only_density():
    def unavailable_inverse(bijection, y, condition):
        raise AssertionError("Sampling must not evaluate the numerical inverse.")

    component = Transformed(
        Normal(jnp.zeros(1)),
        NumericalInverse(Affine(loc=jnp.ones(1), scale=2), unavailable_inverse),
    )
    dist = Concatenate([component, Exponential(jnp.full(2, 3))])
    samples, log_probs = eqx.filter_jit(dist.sample_and_log_prob)(jr.key(5), (4,))
    reference = norm.logpdf(samples[:, 0], loc=1, scale=2)
    reference += expon.logpdf(samples[:, 1:], scale=1 / 3).sum(-1)
    np.testing.assert_allclose(log_probs, reference, rtol=1e-5)


def test_concatenate_validation():
    with pytest.raises(ValueError, match="at least one"):
        Concatenate([])
    with pytest.raises(ValueError, match="scalar"):
        Concatenate([Normal()])
    with pytest.raises(ValueError, match="same event rank"):
        Concatenate([Normal(jnp.zeros(1)), Normal()])
    with pytest.raises(ValueError, match="Expected shapes to match"):
        Concatenate([Normal(jnp.zeros((2, 3))), Normal(jnp.zeros((2, 4)))])
    for axis in (-2, 1):
        with pytest.raises(ValueError, match="out of bounds"):
            Concatenate([Normal(jnp.zeros(1))], axis)
