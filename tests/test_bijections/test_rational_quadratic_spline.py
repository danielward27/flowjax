import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import vmap
from jax.tree_util import tree_leaves, tree_map

from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import StandardNormal, Transformed


@pytest.mark.parametrize("interval", [3, (-4, 5)])
def test_RationalQuadraticSpline_tails(interval):
    key = jr.key(0)
    spline = RationalQuadraticSpline(knots=10, interval=interval)

    # Change to random initialisation, rather than identity.
    spline = tree_map(
        lambda x: jr.normal(key, x.shape) if eqx.is_inexact_array(x) else x,
        spline,
    )

    x = jr.uniform(key, (5,), minval=spline.interval[0], maxval=spline.interval[1])
    y = vmap(spline.transform)(x)
    assert pytest.approx(x, abs=1e-5) != y

    # Outside interval, default to identity
    x = jnp.array([spline.interval[0] - 1, spline.interval[1] + 1])
    y = vmap(spline.transform)(x)
    assert pytest.approx(x, abs=1e-5) == y


@pytest.mark.parametrize("interval", [3, (-4, 5)])
def test_RationalQuadraticSpline_init(interval):
    # Test it is initialized at the identity
    x = jnp.array([-7, 0.1, 2, 1])
    spline = RationalQuadraticSpline(knots=10, interval=interval)
    y = vmap(spline.transform)(x)
    assert pytest.approx(x, abs=1e-6) == y


def test_RationalQuadraticSpline_out_of_bounds_grad_nan():
    """Checks that gradients do not contain NaNs for out of interval inputs."""
    key = jr.key(1)
    spline = RationalQuadraticSpline(knots=4, interval=1.0)
    spline = tree_map(  # Not identity
        lambda x: x + jr.normal(key, x.shape) * 1.5 if eqx.is_inexact_array(x) else x,
        spline,
    )
    val = jnp.array(50.0)  # Out of bounds
    forward_grad = eqx.filter_grad(lambda m, v: m.transform_and_log_det(v)[0].sum())(
        spline, val
    )
    inverse_grad = eqx.filter_grad(lambda m, v: m.inverse_and_log_det(v)[0].sum())(
        spline, val
    )

    for grad in (forward_grad, inverse_grad):
        assert not any(
            jnp.any(jnp.isnan(leaf)) for leaf in tree_leaves(grad) if eqx.is_array(leaf)
        ), "NaNs detected in gradients for out-of-bounds pass"


@pytest.mark.parametrize("interval", [3, (-4, 5)])
@pytest.mark.parametrize("differentiate", [jax.jacfwd, jax.jacrev])
def test_RationalQuadraticSpline_identity_boundary_derivatives(interval, differentiate):
    spline = RationalQuadraticSpline(knots=4, interval=interval)
    points = jnp.array(spline.interval, dtype=float)
    for fn in (spline.transform, spline.inverse):
        assert jnp.allclose(jax.vmap(fn)(points), points)
        derivative = jax.jit(jax.vmap(differentiate(fn)))(points)
        assert jnp.allclose(derivative, jnp.ones_like(points), atol=1e-6)


@pytest.mark.parametrize("interval", [3, (-4, 5)])
def test_RationalQuadraticSpline_boundary_log_det_matches_derivative(interval):
    spline = RationalQuadraticSpline(knots=4, interval=interval)
    derivatives = jnp.array([1.5, 1.0, 1.0, 1.0, 1.0, 2.0])
    spline = eqx.tree_at(lambda s: s.derivatives, spline, derivatives)
    points = jnp.array(spline.interval, dtype=float)
    for fn, expected in (
        (spline.transform_and_log_det, derivatives[jnp.array([0, -1])]),
        (spline.inverse_and_log_det, 1 / derivatives[jnp.array([0, -1])]),
    ):
        values, log_det = jax.jit(jax.vmap(fn))(points)
        actual = jax.jit(jax.vmap(jax.grad(lambda x, fn=fn: fn(x)[0])))(points)
        assert jnp.allclose(values, points)
        assert jnp.allclose(actual, expected, atol=1e-6)
        assert jnp.allclose(log_det, jnp.log(expected), atol=1e-6)


@pytest.mark.parametrize("interval", [3, (-4, 5)])
def test_RationalQuadraticSpline_identity_normal_score_at_boundary(interval):
    spline = RationalQuadraticSpline(knots=4, interval=interval)
    dist = Transformed(StandardNormal(), spline)
    points = jnp.array(spline.interval, dtype=float)
    score = jax.jit(jax.grad(lambda x: dist.log_prob(x).sum()))(points)
    assert jnp.allclose(score, -points, atol=1e-5)
