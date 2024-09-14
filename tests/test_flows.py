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
KEY = jr.key(0)
KWARGS = {
    "key": jr.key(0),
    "base_dist": StandardNormal((DIM,)),
    "flow_layers": 2,
}


testcases = {
    "BNAF": lambda c: block_neural_autoregressive_flow(**KWARGS, cond_dim=c),
    "triangular_spline_flow": lambda c: triangular_spline_flow(**KWARGS, cond_dim=c),
    "affine_coupling": lambda c: coupling_flow(
        transformer=Affine(), **KWARGS, cond_dim=c
    ),
    "spline_coupling": lambda c: coupling_flow(
        transformer=RationalQuadraticSpline(knots=3, interval=2),
        cond_dim=c,
        **KWARGS,
    ),
    "affine_masked_autoregessive": lambda c: masked_autoregressive_flow(
        transformer=Affine(), **KWARGS, cond_dim=c
    ),
    "planar": lambda c: planar_flow(**KWARGS, cond_dim=c, width_size=5, depth=1),
}


@pytest.mark.parametrize("flowname", testcases.keys())
@pytest.mark.parametrize("cond_dim", [None, 2])
def test_flow_sample(flowname, cond_dim):
    flow = testcases[flowname](cond_dim)
    cond = None if flow.cond_shape is None else jr.normal(KEY, flow.cond_shape)
    try:
        assert flow._sample(KEY, cond).shape == flow.shape
    except NotImplementedError:
        pass


@pytest.mark.parametrize("flowname", testcases.keys())
@pytest.mark.parametrize("cond_dim", [None, 2])
def test_flow_log_prob(flowname, cond_dim):
    flow = testcases[flowname](cond_dim)
    x = jr.normal(KEY, flow.shape)
    cond = None if flow.cond_shape is None else jr.normal(KEY, flow.cond_shape)
    assert flow._log_prob(x, cond).shape == ()
