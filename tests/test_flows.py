import jax.random as jr
import pytest

from flowjax.bijections import Affine, RationalQuadraticSpline
from flowjax.distributions import StandardNormal
from flowjax.flows import (
    block_neural_autoregressive_flow,
    coupling_flow,
    masked_autoregressive_flow,
    planar_flow,
    triangular_spline_flow,
)

DIM = 3
KEY = jr.PRNGKey(0)
KWARGS = {
    "key": jr.PRNGKey(0),
    "base_dist": StandardNormal((DIM,)),
    "flow_layers": 2,
}

testcases = {
    "BNAF": block_neural_autoregressive_flow(**KWARGS),
    "triangular_spline_flow": triangular_spline_flow(**KWARGS),
    "Affine_Coupling": coupling_flow(transformer=Affine(), **KWARGS),
    "Spline_Coupling": coupling_flow(
        transformer=RationalQuadraticSpline(knots=3, interval=2),
        **KWARGS,
    ),
    "Affine_MaskedAutoregessive": masked_autoregressive_flow(
        transformer=Affine(),
        **KWARGS,
    ),
    "Planar": planar_flow(**KWARGS),
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
    "BNAF": block_neural_autoregressive_flow(**KWARGS, cond_dim=2),
    "triangular_spline_flow": triangular_spline_flow(**KWARGS, cond_dim=2),
    "Affine_Coupling": coupling_flow(transformer=Affine(), **KWARGS, cond_dim=2),
    "Spline_Coupling": coupling_flow(
        transformer=RationalQuadraticSpline(knots=3, interval=2),
        **KWARGS,
        cond_dim=2,
    ),
    "Affine_MaskedAutoregessive": masked_autoregressive_flow(
        transformer=Affine(),
        **KWARGS,
        cond_dim=2,
    ),
    "Planar": planar_flow(**KWARGS, cond_dim=2, width_size=3, depth=1),
}


@pytest.mark.parametrize(
    "flow",
    conditional_testcases.values(),
    ids=conditional_testcases.keys(),
)
def test_conditional_flow_sample(flow):
    cond = jr.normal(KEY, flow.cond_shape)
    try:
        assert flow._sample(KEY, cond).shape == flow.shape
    except NotImplementedError:
        pass


@pytest.mark.parametrize(
    "flow",
    conditional_testcases.values(),
    ids=conditional_testcases.keys(),
)
def test_conditional_flow_log_prob(flow):
    x = jr.normal(KEY, flow.shape)
    cond = jr.normal(KEY, flow.cond_shape)
    assert flow._log_prob(x, cond).shape == ()
