from jaxflows.flows import Flow
from jaxflows.bijections.permute import Permute
from jaxflows.bijections.coupling import IgnoreCondition
import jax.numpy as jnp
from jax import random
import pytest


def test_Flow():
    bijection = IgnoreCondition(Permute(jnp.array([2, 1, 0])))
    flow = Flow(bijection, 3)
    x = flow.sample(random.PRNGKey(0), n=1)
    assert x.shape == (1, 3)

    x = flow.sample(random.PRNGKey(0), n=2)
    assert x.shape == (2, 3)

    # Note condition is ignored for transformation (but used to infer sample size)
    x = flow.sample(random.PRNGKey(0), condition=jnp.zeros((0,)))
    assert x.shape == (1, 3)

    x = flow.sample(random.PRNGKey(0), condition=jnp.zeros((2, 0)))
    assert x.shape == (2, 3)

    x = flow.sample(random.PRNGKey(0), n=3, condition=jnp.zeros((0,)))
    assert x.shape == (3, 3)
    assert jnp.all(x[0] != x[1])

    # Test works for vector input too
    x1, x2 = x[0], x[None, 0]
    lp1, lp2 = [flow.log_prob(x).item() for x in (x1, x2)]
    assert lp1 == pytest.approx(lp2)
