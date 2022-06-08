from flowjax.flows import Flow, RealNVPFlow, NeuralSplineFlow
from flowjax.bijections.utils import Permute
import jax.numpy as jnp
from jax import random
import pytest


def test_Flow():
    key = random.PRNGKey(0)
    bijection = Permute(jnp.array([2, 1, 0]))
    dim = 3
    flow = Flow(bijection, dim)
    x = flow.sample(key, n=1)
    assert x.shape == (1, dim)

    x = flow.sample(random.PRNGKey(0), n=2)
    assert x.shape == (2, dim)

    # Note condition is ignored for transformation (but can be used to infer sample size)
    x = flow.sample(key, condition=jnp.zeros((0,)), n=5)
    assert x.shape == (5, dim)

    x = flow.sample(key, condition=jnp.zeros((5, 0)))
    assert x.shape == (5, dim)

    with pytest.raises(AssertionError):
        flow.sample(key, condition=jnp.zeros((5, 0)), n=3)

    with pytest.raises(AssertionError):
        flow.sample(key, condition=jnp.zeros((0,)))

    # Test log prob work for vector and matrices input too
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
