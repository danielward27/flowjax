# %%
import jax.numpy as jnp
from jax import random 
from realnvp.bijections.rational_quadratic_spline import RationalQuadraticSpline1D, RationalQuadraticSpline
import pytest



# %%

def test_RationalQuadraticSpline1D():
    spline = RationalQuadraticSpline1D(K=5, B=3)
    params = random.normal(random.PRNGKey(0), (spline.num_params(),))
    transform_args = spline.get_args(params)

    x = jnp.array(0.4)
    y = spline.transform(x, *transform_args)
    x_reconstructed = spline.inverse(y, *transform_args)
    assert x == pytest.approx(x_reconstructed)
    assert (x != y).all()

    # Test identity padding
    for x_val in [-20, 20]:
        x = jnp.array(x_val)
        y = spline.transform(x, *transform_args)
        x_reconstructed = spline.inverse(y, *transform_args)
        assert x == pytest.approx(y)
        assert y == pytest.approx(x_reconstructed)


# # %% plot to see which is wrong
# x = jnp.linspace(-6, 6)
# y = [spline.transform(xi, *args).item() for xi in x]
# y = jnp.array(y)

# import matplotlib.pyplot as plt
# plt.plot(x,y)

# x_reconstructed = [spline.inverse(yi, *args).item() for yi in y]

# plt.plot(x_reconstructed, y)


# %%

# x_dim = 5
# params = random.normal(key, (spline.num_params(x_dim), ))
# key, subkey = random.split(key)
# x = random.normal(key, (x_dim,))

# # %%
# args = spline.get_args(params)
# print(x)
# y = spline.transform(x, *args)

# print(y)

# x_reconstructed = spline.inverse(x, *args)
# print(x_reconstructed)


# # %%

# import pytest
# import jax.numpy as jnp
# from jax import random
# from realnvp.bijections import (
#     Affine,
#     Permute,
#     Coupling,
#     RealNVP,
#     RationalQuadraticSpline,
# )
# from realnvp.bijections.coupling import Coupling
# from jax import random






# # %%
# import pytest
# import jax.numpy as jnp
# from jax import random
# from realnvp.bijections import (
#     Affine,
#     Permute,
#     Coupling,
#     RealNVP,
#     RationalQuadraticSpline,
# )
# from jax import random

# spline = RationalQuadraticSpline(K=5, B=3)
# x = jnp.array([-20])
# params = random.normal(random.PRNGKey(0), (spline.num_params(),))
# transform_args = spline.get_args(params)
# y = spline.transform(x, *transform_args)
# x_reconstructed = spline.inverse(y, *transform_args)
# # assert x == pytest.approx(x_reconstructed)

# # %%

# import pytest
# import jax.numpy as jnp
# from jax import random
# from realnvp.bijections import (
#     Affine,
#     Permute,
#     Coupling,
#     RealNVP,
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
