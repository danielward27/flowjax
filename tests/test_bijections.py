import pytest
import jax.numpy as jnp
from jax import random
from realnvp.bijections import Affine, Permute, CouplingLayer, RealNVP, RationalQuadraticSpline
from jax import random

def test_Affine():
    d = 5
    b = Affine()
    x = jnp.arange(5)
    loc, scale = jnp.ones(d), jnp.ones(d)
    y = b.transform(x, loc, scale)
    x_reconstructed = b.inverse(y, loc, scale)
    assert x == pytest.approx(x_reconstructed)
    assert (x != y).all()

def test_Permute():
    x = jnp.arange(4)
    permute = Permute(jnp.array([3,2,1,0]))
    y = permute.transform(x)
    x_reconstructed = permute.inverse(y)
    assert x == pytest.approx(x_reconstructed)
    assert (x != y).sum()

def test_CouplingLayer():
    model_key, x_key = random.split(random.PRNGKey(0), 2)
    d=2
    D=5

    coupling_layer = CouplingLayer(
        model_key,
        d = d,
        D = D,
        conditioner_width=10,
        conditioner_depth=3,
        )

    x = random.uniform(x_key, (D,))
    y = coupling_layer(x)[0]

    x_reconstructed = coupling_layer.inverse(y)

    assert x == pytest.approx(x_reconstructed)
    assert x[:d] == pytest.approx(y[:d])
    assert (x[d:] != y[d:]).all()

def test_RealNVP():
    model_key, x_key = random.split(random.PRNGKey(0), 2)
    D = 5
    model = RealNVP(model_key, D, 10, 2, 3)

    x = random.uniform(x_key, (D,))
    z = model.transform(x)
    x_reconstructed = model.inverse(z)

    assert x == pytest.approx(x_reconstructed)


def test_RationalQuadraticSpline():
    spline = RationalQuadraticSpline(K=5, B=3)
    x = jnp.array(0.4)
    params = random.normal(random.PRNGKey(0), (spline.num_params(),))
    transform_args = spline.get_args(params)
    y = spline.transform(x, *transform_args)
    x_reconstructed = spline.inverse(y, *transform_args)
    assert True


# %% plot transforms and inverses from -B to B and see if looks reasonable
import jax.numpy as jnp
import jax
from jax import random
from realnvp.bijections import RationalQuadraticSpline
from jax import random
import matplotlib.pyplot as plt
from functools import partial
spline = RationalQuadraticSpline(K=5, B=3)

x = jnp.linspace(-spline.B+1e-6, spline.B - 1e-6, 200)
params = random.normal(random.PRNGKey(0), (spline.num_params(),))
transform_args = spline.get_args(params)
y = jax.vmap(spline.transform, in_axes=(0, None, None, None))(x, *transform_args)
plt.plot(x,y)
# %%
# inverse
x_reconstructed = jax.vmap(spline.inverse, in_axes=(0, None, None, None))(y, *transform_args)

# %%
plt.plot(x,y)
plt.plot(x_reconstructed, y)
# inverse is broken

#define method that just takes array for ParameterisedBijection and outputs 

# %%
spline.inverse(jnp.array(0.3), *transform_args)

# %%

def tuple_output_test(a):
    return (a, a)

a = jnp.array([1, 2, 3])
jax.vmap(tuple_output_test)(a)
# tuple outputs seem to work fine.
# %%
