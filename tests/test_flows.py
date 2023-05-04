import jax.random as jr
import pytest

from flowjax.bijections import Affine, RationalQuadraticSpline
from flowjax.distributions import StandardNormal
from flowjax.flows import (
    BlockNeuralAutoregressiveFlow,
    CouplingFlow,
    MaskedAutoregressiveFlow,
    TriangularSplineFlow,
)

DIM = 3
KEY = jr.PRNGKey(0)
KWARGS = {
    "key": jr.PRNGKey(0),
    "base_dist": StandardNormal((DIM,)),
    "flow_layers": 2,
}

testcases = {
    "BNAF": BlockNeuralAutoregressiveFlow(**KWARGS),
    "TriangularSplineFlow": TriangularSplineFlow(**KWARGS),
    "Affine_Coupling": CouplingFlow(transformer=Affine(), **KWARGS),
    "Spline_Coupling": CouplingFlow(
        transformer=RationalQuadraticSpline(3, 2), **KWARGS
    ),
    "Affine_MaskedAutoregessive": MaskedAutoregressiveFlow(
        transformer=Affine(), **KWARGS
    ),
}


@pytest.mark.parametrize("flow", testcases.values(), ids=testcases.keys())
def test_unconditional_flow_sample(flow):
    try:
        assert flow._sample(KEY).shape == flow.shape
    except NotImplementedError:
        pass


@pytest.mark.parametrize("flow", testcases.values(), ids=testcases.keys())
def test_unconditional_flow_log_prob(flow):
    x = jr.normal(KEY, flow.shape)
    assert flow._log_prob(x).shape == ()


conditional_testcases = {
    "BNAF": BlockNeuralAutoregressiveFlow(**KWARGS, cond_dim=2),
    "TriangularSplineFlow": TriangularSplineFlow(**KWARGS, cond_dim=2),
    "Affine_Coupling": CouplingFlow(transformer=Affine(), **KWARGS, cond_dim=2),
    "Spline_Coupling": CouplingFlow(
        transformer=RationalQuadraticSpline(3, 2), **KWARGS, cond_dim=2
    ),
    "Affine_MaskedAutoregessive": MaskedAutoregressiveFlow(
        transformer=Affine(), **KWARGS, cond_dim=2
    ),
}


@pytest.mark.parametrize(
    "flow", conditional_testcases.values(), ids=conditional_testcases.keys()
)
def test_conditional_flow_sample(flow):
    cond = jr.normal(KEY, flow.cond_shape)
    try:
        assert flow._sample(KEY, cond).shape == flow.shape
    except NotImplementedError:
        pass


@pytest.mark.parametrize(
    "flow", conditional_testcases.values(), ids=conditional_testcases.keys()
)
def test_conditional_flow_log_prob(flow):
    x = jr.normal(KEY, flow.shape)
    cond = jr.normal(KEY, flow.cond_shape)
    assert flow._log_prob(x, cond).shape == ()
