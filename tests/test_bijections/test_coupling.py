import pytest
from jax import random
from flowjax.bijections.coupling import Coupling
from flowjax.bijections.parameterised import Affine


def test_Coupling():
    model_key, x_key = random.split(random.PRNGKey(0), 2)
    d = 2
    D = 5

    coupling = Coupling(
        model_key, Affine(), d=d, D=D, cond_dim=0, nn_width=10, nn_depth=3
    )

    x = random.uniform(x_key, (D,))

    y = coupling.transform(x)

    x_reconstructed = coupling.inverse(y)

    assert x == pytest.approx(x_reconstructed)
    assert x[:d] == pytest.approx(y[:d])
    assert (x[d:] != y[d:]).all()
