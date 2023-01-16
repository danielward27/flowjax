import pytest
import jax.random as jr

from flowjax.distributions import StandardNormal
from flowjax.flows import (
    BlockNeuralAutoregressiveFlow,
    CouplingFlow,
    MaskedAutoregressiveFlow,
    TriangularSplineFlow,
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
    "Spline_Coupling": CouplingFlow(
        transformer=RationalQuadraticSpline(3, 2), **kwargs
    ),
    "Affine_MaskedAutoregessive": MaskedAutoregressiveFlow(
        transformer=Affine(), **kwargs
        ),
}



@pytest.mark.parametrize("flow", testcases.values(), ids=testcases.keys())
def test_unconditional_flow_sample(flow):
    try:
        assert flow._sample(key).shape == flow.shape
    except NotImplementedError:
        pass


@pytest.mark.parametrize("flow", testcases.values(), ids=testcases.keys())
def test_unconditional_flow_log_prob(flow):
    x = jr.normal(key, flow.shape)
    assert flow._log_prob(x).shape == ()




conditional_testcases = {
    "BNAF": BlockNeuralAutoregressiveFlow(**kwargs, cond_dim=2),
    "TriangularSplineFlow": TriangularSplineFlow(**kwargs, cond_dim=2),
    "Affine_Coupling": CouplingFlow(transformer=Affine(), **kwargs, cond_dim=2),
    "Spline_Coupling": CouplingFlow(
        transformer=RationalQuadraticSpline(3, 2), **kwargs, cond_dim=2
    ),
    "Affine_MaskedAutoregessive": MaskedAutoregressiveFlow(
        transformer=Affine(), **kwargs, cond_dim=2
        ),
}


@pytest.mark.parametrize("flow", conditional_testcases.values(), ids=conditional_testcases.keys())
def test_conditional_flow_sample(flow):
    cond = jr.normal(key, flow.cond_shape)
    try:
        assert flow._sample(key, cond).shape == flow.shape
    except NotImplementedError:
        pass


@pytest.mark.parametrize("flow", conditional_testcases.values(), ids=conditional_testcases.keys())
def test_conditional_flow_log_prob(flow):
    x = jr.normal(key, flow.shape)
    cond = jr.normal(key, flow.cond_shape)
    assert flow._log_prob(x, cond).shape == ()
    
