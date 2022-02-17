import pytest
import jax.numpy as jnp
from jax import random
from realnvp.bijections import Affine, Permute, CouplingLayer, RealNVP
import jax.numpy as jnp
from jax import random
import pytest

def test_Affine():
    d = 5
    b = Affine()
    x = jnp.arange(5)
    params = jnp.ones((d*2),)

    y = b.transform(x, params)
    x_reconstructed = b.inverse(y, params)
    assert pytest.approx(x, x_reconstructed)
    assert (x != y).all()

def test_Permute():
    x = jnp.arange(4)
    permute = Permute(jnp.array([3,2,1,0]))
    y = permute.transform(x)
    x_reconstructed = permute.inverse(y)
    assert pytest.approx(x, x_reconstructed)
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

    assert pytest.approx(x, x_reconstructed)
    assert pytest.approx(x[:d], y[:d])
    assert (x[d:] != y[d:]).all()

def test_RealNVP():
    model_key, x_key = random.split(random.PRNGKey(0), 2)
    D = 5
    model = RealNVP(model_key, D, 10, 2, 3)

    x = random.uniform(x_key, (D,))
    z = model.transform(x)
    x_reconstructed = model.inverse(z)

    assert pytest.approx(x, x_reconstructed)

    