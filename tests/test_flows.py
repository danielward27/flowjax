from jaxflows.flows import Flow
from jaxflows.bijections.permute import Permute
from jaxflows.bijections.coupling import IgnoreCondition
import jax.numpy as jnp
from jax import random
import pytest


def test_Flow():
    bijection = IgnoreCondition(Permute(jnp.array([2, 1, 0])))
    flow = Flow(bijection, 3)
    x = flow.sample(random.PRNGKey(0), 10)
    lp = flow.log_prob(x)

    assert x.shape == (10, 3)
    assert lp.shape == (10,)

    # Test broadcasting
    x1 = x[None, 0]
    x2 = x[0]
    lp1, lp2 = [flow.log_prob(x).item() for x in (x1, x2)]
    assert lp1 == pytest.approx(lp2)

