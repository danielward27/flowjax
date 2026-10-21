from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from flowjax.bijections import AdditiveCondition, NumericalInverse
from flowjax.root_finding import (
    elementwise_autoregressive_bisection,
    root_finder_to_inverter,
)


class ScaledCondition(eqx.Module):
    scale: jax.Array

    def __call__(self, condition):
        return self.scale * condition


@pytest.mark.parametrize("differentiate", [jax.grad, jax.jacfwd, jax.jacrev])
def test_numerical_inverse_differentiates_condition_and_parameters(differentiate):
    inverter = root_finder_to_inverter(
        partial(
            elementwise_autoregressive_bisection,
            lower=jnp.full(2, -5.0),
            upper=jnp.full(2, 5.0),
            atol=1e-6,
        )
    )

    def inverse_total(y, condition, scale):
        bijection = AdditiveCondition(ScaledCondition(scale), (2,), (2,))
        inverse = NumericalInverse(bijection, inverter)
        return inverse.inverse(y, condition).sum()

    y, condition, scale = jnp.array([2.0, 1.0]), jnp.array([0.5, -0.25]), jnp.array(2.0)
    derivative = jax.jit(differentiate(inverse_total, argnums=(0, 1, 2)))
    dy, dcondition, dscale = derivative(y, condition, scale)

    assert jnp.allclose(inverse_total(y, condition, scale), 2.5, atol=1e-5)
    assert jnp.allclose(dy, jnp.ones(2))
    assert jnp.allclose(dcondition, jnp.full(2, -2.0))
    assert jnp.allclose(dscale, -0.25)


def test_nonimplicit_inverse_keeps_keyword_only_condition_support():
    def inverter(bijection, y, *, condition):
        return bijection.inverse(y, condition)

    bijection = AdditiveCondition(ScaledCondition(jnp.array(2.0)), (), ())
    inverse = NumericalInverse(bijection, inverter, use_implicit_differentation=False)
    derivative = jax.grad(lambda condition: inverse.inverse(jnp.array(2.0), condition))

    assert jnp.allclose(derivative(jnp.array(0.5)), -2.0)


def test_implicit_inverse_keeps_positional_inverter_support():
    def inverter(bijection, y, c):
        return bijection.inverse(y, c)

    bijection = AdditiveCondition(ScaledCondition(jnp.array(2.0)), (), ())
    inverse = NumericalInverse(bijection, inverter)

    assert jnp.allclose(inverse.inverse(jnp.array(2.0), jnp.array(0.5)), 1.0)
