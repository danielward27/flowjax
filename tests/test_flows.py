from typing import Any, Dict

import jax.numpy as jnp
import pytest
import jax.random as jr


from flowjax.distributions import StandardNormal
from flowjax.flows import (
    BlockNeuralAutoregressiveFlow,
    CouplingFlow,
    MaskedAutoregressiveFlow,
    TriangularSplineFlow
)

from flowjax.bijections import Affine, RationalQuadraticSpline

dim = 3
key = jr.PRNGKey(0)
kwargs = {
    "key": jr.PRNGKey(0),
    "base_dist": StandardNormal((dim,)),
    "flow_layers": 2,
}

testcases = {
    "BNAF": BlockNeuralAutoregressiveFlow(**kwargs),
    "TriangularSplineFlow": TriangularSplineFlow(**kwargs),
    "Affine_Coupling": CouplingFlow(transformer=Affine(), **kwargs),
    "Spline_Coupling": CouplingFlow(transformer=RationalQuadraticSpline(3,2), **kwargs),
    
    # "Affine_MaskedAutoregessive": MaskedAutoregressiveFlow(transformer=Affine(), **kwargs),
    #     {"transformer": } | common_kwargs,
    # ),
    
}



@pytest.mark.parametrize("flow", testcases.values(), ids=testcases.keys())
def test_unconditional_flow_sample(flow):
    n = 5
    try:
        assert flow.sample(key, sample_shape=(n,)).shape == (n, ) + flow.shape
        assert flow.sample(key).shape == flow.shape
    except NotImplementedError:
        pass


@pytest.mark.parametrize("flow", testcases.values(), ids=testcases.keys())
def test_unconditional_flow_log_prob(flow):
    n = 5
    x = jr.normal(key, (n, ) + flow.shape)
    lp = flow.log_prob(x)
    assert lp.shape == (n,)

    lp2 = flow.log_prob(x[0])
    assert lp[0] == pytest.approx(lp2, abs=1e-5)

    # Test wrong dimensions raises error
    with pytest.raises(ValueError):
        flow.log_prob(jnp.ones((n, flow.shape[0] + 1)))

    with pytest.raises(ValueError):
        flow.log_prob(jnp.ones((flow.shape[0] + 1,)))


# cond_testcases = {n: t(**kwargs, cond_dim=2) for n, t, kwargs in testcases}


# @pytest.mark.parametrize("flow", cond_testcases.values(), ids=cond_testcases.keys())
# def test_conditional_flow_sample(flow):
#     n = 5
#     key = jr.PRNGKey(0)
#     cond_1d = jnp.ones(flow.cond_dim)
#     cond_2d = jnp.ones((n, flow.cond_dim))

#     try:
#         x = flow.sample(key, condition=cond_1d)
#         assert x.shape == (flow.dim,)

#         x = flow.sample(key, condition=cond_1d, n=n)
#         assert x.shape == (n, flow.dim)

#         x = flow.sample(key, condition=cond_2d)
#         assert x.shape == (n, flow.dim)

#         with pytest.raises(ValueError):
#             x = flow.sample(key, condition=cond_2d, n=n)

#     except NotImplementedError:
#         pass


# @pytest.mark.parametrize("flow", cond_testcases.values(), ids=cond_testcases.keys())
# def test_conditional_flow_log_prob(flow):
#     n = 5

#     x_1d = jnp.ones(flow.dim)
#     x_2d = jnp.ones((n, flow.dim))

#     cond_1d = jnp.ones(flow.cond_dim)
#     cond_2d = jnp.ones((n, flow.cond_dim))

#     assert flow.log_prob(x_1d, cond_1d).shape == ()
#     assert flow.log_prob(x_1d, cond_2d).shape == (n,)
#     assert flow.log_prob(x_2d, cond_2d).shape == (n,)
#     assert flow.log_prob(x_2d, cond_1d).shape == (n,)

#     with pytest.raises(ValueError):
#         flow.log_prob(x_1d, condition=jnp.ones(flow.cond_dim + 1))

#     with pytest.raises(ValueError):
#         flow.log_prob(x_1d, condition=jnp.ones((n, flow.cond_dim + 1)))
