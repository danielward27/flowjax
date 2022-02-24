import pytest
from jax import random
from jaxflows.bijections.coupling import Coupling, CouplingStack
from jaxflows.bijections.affine import Affine


def test_Coupling():
    model_key, x_key = random.split(random.PRNGKey(0), 2)
    d = 2
    D = 5

    coupling = Coupling(
        model_key, Affine(), d=d, D=D, conditioner_width=10, conditioner_depth=3,
    )

    x = random.uniform(x_key, (D,))
    y = coupling(x)[0]

    x_reconstructed = coupling.inverse(y)

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
