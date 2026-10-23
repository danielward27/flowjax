import math

import jax
import jax.numpy as jnp
import pytest

from flowjax.bijections import LeakyTanh
from flowjax.distributions import StandardNormal, Transformed


def _inverse_reference(y, max_val):
    slope = 1 / math.cosh(max_val) ** 2
    intercept = math.tanh(max_val) - slope * max_val
    if abs(y) >= math.tanh(max_val):
        x = (y - math.copysign(intercept, y)) / slope
        return x, 1 / slope, -x / slope
    x = math.atanh(y)
    derivative = 1 / (1 - y**2)
    return x, derivative, (-x + 2 * y) * derivative


@pytest.mark.parametrize("max_val", [1.0, 2.0])
@pytest.mark.parametrize("differentiate", [jax.jacfwd, jax.jacrev])
def test_leaky_tanh_inverse_derivative_at_unit_values(max_val, differentiate):
    y = jnp.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    bijection = LeakyTanh(max_val, y.shape)
    reference = jnp.array([_inverse_reference(float(v), max_val) for v in y])

    x, log_det = jax.jit(bijection.inverse_and_log_det)(y)
    derivative = jax.jit(differentiate(bijection.inverse))(y)

    assert jnp.allclose(x, reference[:, 0])
    assert jnp.allclose(log_det, jnp.log(reference[:, 1]).sum())
    assert jnp.allclose(bijection.transform(x), y)
    assert jnp.allclose(derivative, jnp.diag(reference[:, 1]))


@pytest.mark.parametrize("max_val", [1.0, 2.0])
def test_leaky_tanh_transformed_normal_score_at_unit_values(max_val):
    y = jnp.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    dist = Transformed(StandardNormal(), LeakyTanh(max_val))
    expected = jnp.array([_inverse_reference(float(v), max_val)[2] for v in y])
    score = jax.jit(jax.grad(lambda values: dist.log_prob(values).sum()))(y)

    assert jnp.isfinite(dist.log_prob(y)).all()
    assert jnp.allclose(score, expected)
