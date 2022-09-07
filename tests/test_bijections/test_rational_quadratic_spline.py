import pytest
from flowjax.bijections.transformers import RationalQuadraticSpline
from jax import random
import jax.numpy as jnp

def test_RationalQuadraticSpline_tails():
    spline = RationalQuadraticSpline(K=5, B=3)
    x = jnp.array([-20, 0.1, 2, 20])
    params = random.normal(random.PRNGKey(0), (spline.num_params(x.shape[0]),))
    transform_args = spline.get_args(params)
    y = spline.transform(x, *transform_args)
    expected_changed = jnp.array([True, False, False, True])  # identity padding
    assert ((jnp.abs((y - x)) <= 1e-5) == expected_changed).all()
