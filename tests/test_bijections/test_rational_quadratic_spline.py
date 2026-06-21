import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import vmap
from jax.tree_util import tree_leaves, tree_map

from flowjax.bijections import RationalQuadraticSpline


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
