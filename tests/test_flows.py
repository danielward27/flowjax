from flowjax.flows import Flow, RealNVPFlow, NeuralSplineFlow, BlockNeuralAutoregressiveFlow
from flowjax.bijections.utils import Permute
from flowjax.distributions import Normal
import jax.numpy as jnp
from jax import random
import pytest

def test_unconditional_Flow():
    key = random.PRNGKey(0)
    bijection = Permute(jnp.array([2, 1, 0]))
    dim = 3
    flow = Flow(Normal(dim), bijection)
    x = flow.sample(key, n=1)
    assert x.shape == (1, dim)

    x = flow.sample(random.PRNGKey(0), n=2)
    assert x.shape == (2, dim)

    # Test log prob work for vector and matrices input
    x1, x2 = x[0], x[None, 0]
    lp1, lp2 = [flow.log_prob(x).item() for x in (x1, x2)]
    assert lp1 == pytest.approx(lp2)

    # Test wrong dimensions
    with pytest.raises(ValueError):
        flow.log_prob(jnp.ones((5,5)))


def test_RealNVPFlow():
    key = random.PRNGKey(1)
    flow = RealNVPFlow(key, Normal(3), num_layers=2)
    x = flow.sample(key, n=10)
    assert x.shape == (10, 3)

    lp = flow.log_prob(x)
    assert lp.shape == (10,)

    lp2 = flow.log_prob(x[0])
    assert lp[0] == pytest.approx(lp2) 

# def test_Flow():
#     flow=1
#     # Note condition is ignored for transformation (but can be used to infer sample size)
#     x = flow.sample(key, condition=jnp.zeros((0,)), n=5)
#     assert x.shape == (5, dim)

#     x = flow.sample(key, condition=jnp.zeros((5, 0)))
#     assert x.shape == (5, dim)

#     with pytest.raises(AssertionError):
#         flow.sample(key, condition=jnp.zeros((5, 0)), n=3)

#     with pytest.raises(AssertionError):
#         flow.sample(key, condition=jnp.zeros((0,)))


def test_NeuralSplineFlow():
    # Unconditional
    n = 10
    dim = 3
    key = random.PRNGKey(2)
    flow = NeuralSplineFlow(key, Normal(dim), num_layers=2)
    x = flow.sample(key, n=n)
    assert x.shape == (n, dim)

    lp = flow.log_prob(x)
    assert lp.shape == (n,)

    # Conditional
    cond_dim = 2
    flow = NeuralSplineFlow(key, Normal(dim), cond_dim=cond_dim, num_layers=2)
    cond = random.uniform(key, (n, cond_dim))
    x = flow.sample(key, condition=cond)
    lp = flow.log_prob(x, cond)
    assert lp.shape == (n,)

    lp = flow.log_prob(x, jnp.ones(cond_dim))
    assert lp.shape == (n,)

    lp = flow.log_prob(jnp.ones(dim), cond)
    assert lp.shape == (n,)

    x = flow.sample(key, condition=jnp.ones(2), n=n)
    assert x.shape == (n, dim)


def test_BlockNeuralAutoregressiveFlow():
    key = random.PRNGKey(1)
    dim, n, cond_dim = 3, 10, 2
    flow = BlockNeuralAutoregressiveFlow(key, Normal(dim), flow_layers=2)
    x = random.uniform(key, (n, dim))
    lps = flow.log_prob(x)
    assert lps.shape == (n,)

    flow = BlockNeuralAutoregressiveFlow(key, Normal(dim), cond_dim=cond_dim, flow_layers=2)
    x = random.uniform(key, (n, dim))
    cond = random.normal(key, (n, cond_dim))
    lps = flow.log_prob(x, cond)
    assert lps.shape == (n,)