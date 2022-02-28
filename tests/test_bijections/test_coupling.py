import pytest
from jax import random
from jaxflows.bijections.coupling import Coupling, CouplingStack
from jaxflows.bijections.affine import Affine
import jax.numpy as jnp


def test_Coupling():
    model_key, x_key = random.split(random.PRNGKey(0), 2)
    d = 2
    D = 5

    coupling = Coupling(model_key, Affine(), d=d, D=D, nn_width=10, nn_depth=3,)

    x = random.uniform(x_key, (D,))

    condition = jnp.zeros((x.shape[0], 0))  # No condition
    y = coupling.transform(x, condition)[0]

    x_reconstructed = coupling.inverse(y, condition)

    assert x == pytest.approx(x_reconstructed)
    assert x[:d] == pytest.approx(y[:d])
    assert (x[d:] != y[d:]).all()


def test_CouplingStack():
    model_key, x_key = random.split(random.PRNGKey(0), 2)
    D = 5
    model = CouplingStack(model_key, Affine(), D, 10, 2, 3)

    x = random.uniform(x_key, (D,))
    z = model.transform(x)
    x_reconstructed = model.inverse(z)
    assert x == pytest.approx(x_reconstructed)
