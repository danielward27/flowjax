from flowjax.flows import Flow, RealNVPFlow, NeuralSplineFlow
from flowjax.bijections.utils import Permute
import jax.numpy as jnp
from jax import random
import pytest


def test_Flow():
    key = random.PRNGKey(0)
    bijection = Permute(jnp.array([2, 1, 0]))
    flow = Flow(bijection, 3)
    x = flow.sample(key, n=1)
    assert x.shape == (1, 3)

    x = flow.sample(random.PRNGKey(0), n=2)
    assert x.shape == (2, 3)

    # Note condition is ignored for transformation (but used to infer sample size)
    x = flow.sample(key, condition=jnp.zeros((0,)))
    assert x.shape == (1, 3)

    x = flow.sample(key, condition=jnp.zeros((2, 0)))
    assert x.shape == (2, 3)

    x = flow.sample(key, n=3, condition=jnp.zeros((0,)))
    assert x.shape == (3, 3)
    assert jnp.all(x[0] != x[1])

    # Test works for vector input too
    x1, x2 = x[0], x[None, 0]
    lp1, lp2 = [flow.log_prob(x).item() for x in (x1, x2)]
    assert lp1 == pytest.approx(lp2)


def test_RealNVPFlow():
    key = random.PRNGKey(1)
    flow = RealNVPFlow(key, 3)
    x = flow.sample(key, n=10)
    assert x.shape == (10, 3)

    lp = flow.log_prob(x)
    assert lp.shape == (10,)


def test_NeuralSplineFlow():
    key = random.PRNGKey(2)
    flow = NeuralSplineFlow(key, 3, num_layers=2)
    x = flow.sample(key, n=10)
    assert x.shape == (10, 3)

    lp = flow.log_prob(x)
    assert lp.shape == (10,)
