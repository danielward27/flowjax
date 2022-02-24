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
    x = jnp.array([0.4])
    params = random.normal(random.PRNGKey(0), (spline.num_params(),))
    transform_args = spline.get_args(params)
    y = spline.transform(x, *transform_args)
    x_reconstructed = spline.inverse(y, *transform_args)
    assert x ==pytest.approx(x_reconstructed)

