
from jax import random
from realnvp.bijections.real_nvp import RealNVP
import pytest

def test_RealNVP():
    model_key, x_key = random.split(random.PRNGKey(0), 2)
    D = 5
    model = RealNVP(model_key, D, 10, 2, 3)

    x = random.uniform(x_key, (D,))
    z = model.transform(x)
    x_reconstructed = model.inverse(z)

    assert x == pytest.approx(x_reconstructed)