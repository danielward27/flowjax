from typing import Any, Dict

import jax.numpy as jnp
import pytest
from jax import random

from flowjax.bijections.transformers import (
    AffineTransformer,
    RationalQuadraticSplineTransformer,
)
from flowjax.distributions import StandardNormal
from flowjax.flows import (
    BlockNeuralAutoregressiveFlow,
    CouplingFlow,
    MaskedAutoregressiveFlow,
)

dim = 3
common_kwargs = {
    "key": random.PRNGKey(0),
    "base_dist": StandardNormal(dim),
    "flow_layers": 2,
    "nn_width": 10,
    "nn_depth": 1,
}  # type: Dict[str, Any]

testcases = [
    # (name, type, kwargs)}
    (
        "Affine_Coupling",
        CouplingFlow,
        {"transformer": AffineTransformer()} | common_kwargs,
    ),
    (
        "RationalQuadraticSpline_Coupling",
        CouplingFlow,
        {"transformer": RationalQuadraticSplineTransformer(5, 3)} | common_kwargs,
    ),
    (
        "BNAF",
        BlockNeuralAutoregressiveFlow,
        {"key": random.PRNGKey(0), "base_dist": StandardNormal(dim), "flow_layers": 2},
    ),
    (
        "Affine_MaskedAutoregessive",
        MaskedAutoregressiveFlow,
        {"transformer": AffineTransformer()} | common_kwargs,
    ),
]

uncond_testcases = {n: t(**kwargs) for n, t, kwargs in testcases}


@pytest.mark.parametrize("flow", uncond_testcases.values(), ids=uncond_testcases.keys())
def test_unconditional_flow_sample(flow):
    key = random.PRNGKey(0)
    n = 5
    try:
        assert flow.sample(key, n=n).shape == (n, flow.dim)
        assert flow.sample(key).shape == (flow.dim,)
    except NotImplementedError:
        pass


@pytest.mark.parametrize("flow", uncond_testcases.values(), ids=uncond_testcases.keys())
def test_unconditional_flow_log_prob(flow):
    n = 5
    x = random.normal(random.PRNGKey(0), (n, flow.dim))
    lp = flow.log_prob(x)
    assert lp.shape == (n,)

    lp2 = flow.log_prob(x[0])
    assert lp[0] == pytest.approx(lp2, abs=1e-5)

    # Test wrong dimensions raises error
    with pytest.raises(ValueError):
        flow.log_prob(jnp.ones((n, flow.dim + 1)))

    with pytest.raises(ValueError):
        flow.log_prob(jnp.ones((flow.dim + 1,)))


cond_testcases = {n: t(**kwargs, cond_dim=2) for n, t, kwargs in testcases}


@pytest.mark.parametrize("flow", cond_testcases.values(), ids=cond_testcases.keys())
def test_conditional_flow_sample(flow):
    n = 5
    key = random.PRNGKey(0)
    cond_1d = jnp.ones(flow.cond_dim)
    cond_2d = jnp.ones((n, flow.cond_dim))

    try:
        x = flow.sample(key, condition=cond_1d)
        assert x.shape == (flow.dim,)

        x = flow.sample(key, condition=cond_1d, n=n)
        assert x.shape == (n, flow.dim)

        x = flow.sample(key, condition=cond_2d)
        assert x.shape == (n, flow.dim)

        with pytest.raises(ValueError):
            x = flow.sample(key, condition=cond_2d, n=n)

    except NotImplementedError:
        pass


@pytest.mark.parametrize("flow", cond_testcases.values(), ids=cond_testcases.keys())
def test_conditional_flow_log_prob(flow):
    n = 5

    x_1d = jnp.ones(flow.dim)
    x_2d = jnp.ones((n, flow.dim))

    cond_1d = jnp.ones(flow.cond_dim)
    cond_2d = jnp.ones((n, flow.cond_dim))

    assert flow.log_prob(x_1d, cond_1d).shape == ()
    assert flow.log_prob(x_1d, cond_2d).shape == (n,)
    assert flow.log_prob(x_2d, cond_2d).shape == (n,)
    assert flow.log_prob(x_2d, cond_1d).shape == (n,)

    with pytest.raises(ValueError):
        flow.log_prob(x_1d, condition=jnp.ones(flow.cond_dim + 1))

    with pytest.raises(ValueError):
        flow.log_prob(x_1d, condition=jnp.ones((n, flow.cond_dim + 1)))
