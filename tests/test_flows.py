from flowjax.flows import Flow, RealNVPFlow, NeuralSplineFlow, BlockNeuralAutoregressiveFlow
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

def test_broadcast():
    # Matrices
    size_pairs = [((5,2), (5,3)), ((1,2), (5,3)), ((5,2), (1,3)), ((2,), (5,3)), ((5,2), (3,))]
    out_sizes = [((5,2), (5,3))] * len(size_pairs)

    for in_s, out_s in zip(size_pairs, out_sizes):
        a,b = Flow._broadcast(jnp.ones(in_s[0]), jnp.ones(in_s[1]))
        assert (a.shape, b.shape) == out_s


def test_NeuralSplineFlow():
    # Unconditional
    n = 10
    dim = 3
    key = random.PRNGKey(2)
    flow = NeuralSplineFlow(key, dim, num_layers=2)
    x = flow.sample(key, n=n)
    assert x.shape == (n, dim)

    lp = flow.log_prob(x)
    assert lp.shape == (n,)

    # Conditional
    cond_dim = 2
    flow = NeuralSplineFlow(key, dim, condition_dim=cond_dim, num_layers=2)
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

def test_RealNVPFlow():
    key = random.PRNGKey(1)
    flow = RealNVPFlow(key, 3)
    x = flow.sample(key, n=10)
    assert x.shape == (10, 3)

    lp = flow.log_prob(x)
    assert lp.shape == (10,)

def test_BlockNeuralAutoregressiveFlow():
    key = random.PRNGKey(1)
    dim, n, cond_dim = 3, 10, 2
    flow = BlockNeuralAutoregressiveFlow(key, dim, flow_layers=2)
    x = random.uniform(key, (n, dim))
    lps = flow.log_prob(x)
    assert lps.shape == (n,)

    flow = BlockNeuralAutoregressiveFlow(key, dim, condition_dim=cond_dim, flow_layers=2)
    x = random.uniform(key, (n, dim))
    cond = random.normal(key, (n, cond_dim))
    lps = flow.log_prob(x, cond)
    assert lps.shape == (n,)