import jax.numpy as jnp
from jax import random
from flowjax.bijections.rational_quadratic_spline import RationalQuadraticSpline
import pytest


def test_RationalQuadraticSpline():
    spline = RationalQuadraticSpline(K=5, B=3)
    x = jnp.array([-20, 0.1, 2, 20])
    params = random.normal(random.PRNGKey(0), (spline.num_params(x.shape[0]),))
    transform_args = spline.get_args(params)
    y = spline.transform(x, *transform_args)
    x_reconstructed = spline.inverse(y, *transform_args)
    assert x == pytest.approx(x_reconstructed, abs=1e-6)
    expected_changed = jnp.array([True, False, False, True])  # identity padding
    assert ((jnp.abs((y - x)) <= 1e-5) == expected_changed).all()
