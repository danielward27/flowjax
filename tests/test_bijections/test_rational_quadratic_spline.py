import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import vmap
from jax.tree_util import tree_map

from flowjax.bijections import RationalQuadraticSpline
from flowjax.bijections.rational_quadratic_spline import _real_to_increasing_on_interval


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


def test_real_to_increasing_on_interval():
    y = _real_to_increasing_on_interval(jnp.array([-3.0, -4, 5, 0, 0]), (-3, 7))
    assert y.max() == 7
    assert y.min() == -3
    assert jnp.all(jnp.diff(y)) > 0
