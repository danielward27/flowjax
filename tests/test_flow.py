from jaxflows.flow import Flow
from jaxflows.bijections.permute import Permute
from jaxflows.bijections.coupling import IgnoreCondition
import jax.numpy as jnp
from jax import random


def test_Flow():
    bijection = IgnoreCondition(Permute(jnp.array([2, 1, 0])))
    flow = Flow(bijection, 3)
    x = flow.sample(random.PRNGKey(0), 10)
    lp = flow.log_prob(x)

    assert x.shape == (10, 3)
    assert lp.shape == (10,)
