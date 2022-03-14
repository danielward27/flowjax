import jax.numpy as jnp
from jax import random
from flowjax.bijections.rational_quadratic_spline import (
    RationalQuadraticSpline1D,
    RationalQuadraticSpline,
)
import pytest


def test_RationalQuadraticSpline1D():
    spline = RationalQuadraticSpline1D(K=5, B=3)
    params = random.normal(random.PRNGKey(0), (spline.num_params(),))
    transform_args = spline.get_args(params)

    x = jnp.array(0.4)
    y = spline.transform(x, *transform_args)
    x_reconstructed = spline.inverse(y, *transform_args)
    assert x == pytest.approx(x_reconstructed, abs=1e-6)
    assert (x != y).all()

    # Test identity padding
    for x_val in [-20, 20]:
        x = jnp.array(x_val)
        y = spline.transform(x, *transform_args)
        x_reconstructed = spline.inverse(y, *transform_args)
        assert x == pytest.approx(y)
        assert y == pytest.approx(x_reconstructed, abs=1e-6)


def test_RationalQuadraticSpline():
    spline = RationalQuadraticSpline(K=5, B=3)
    x = jnp.array([-20, 0.1, 2, 20])
    params = random.normal(random.PRNGKey(0), (spline.num_params(x.shape[0]),))
    transform_args = spline.get_args(params)
    y = spline.transform(x, *transform_args)
    x_reconstructed = spline.inverse(y, *transform_args)
    assert x == pytest.approx(x_reconstructed, abs=1e-6)
    expected_changed = jnp.array([True, False, False, True])
    assert ((jnp.abs((y - x)) <= 1e-5) == expected_changed).all()


# # %%

# import pytest
# import jax.numpy as jnp
# from jax import random
# from flowjax.bijections import (
#     Affine,
#     Permute,
#     Coupling,
#     flowjax,
#     RationalQuadraticSpline,
# )
# from jax import random
# import jax


# spline = RationalQuadraticSpline(K=5, B=3)
# x = jnp.array([[-20]])
# params = random.normal(random.PRNGKey(0), (spline.num_params(),))

# # %%
# transform_args = spline.get_args(params)
# transform_args = tuple(jnp.expand_dims(arg, 0) for arg in transform_args)
# y = jax.vmap(spline._1d_transform)(x, *transform_args)
# # x_reconstructed = spline.inverse(y, *transform_args)
# # assert x == pytest.approx(x_reconstructed)

# # %%
# # fix tests

# %%
