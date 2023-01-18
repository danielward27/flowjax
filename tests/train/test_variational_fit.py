import pytest

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

from flowjax.train.variational_fit import (
    elbo_loss, 
    variational_fit,
    VariationalLoss, 
    VariationalTarget
)
from flowjax.flows import MaskedAutoregressiveFlow
from flowjax import distributions
from flowjax import bijections

@pytest.mark.parametrize('distribution, target, shape', [
    pytest.param(
        distributions.StandardNormal, 
        distributions.StandardNormal, 
        (2,),
        id='normal_target_normal'
    ),
    pytest.param(
        distributions.StandardNormal, 
        distributions._StandardCauchy, 
        (2,),
        id='normal_target_cauchy'
    ),
    pytest.param(
        distributions.StandardNormal, 
        distributions.StandardNormal, 
        (100,),
        id='higher_dim'
    ),
    pytest.param(
        distributions.StandardNormal, 
        distributions.StandardNormal, 
        (3, 2, 2),
        id='multiple_dim'
    ),
])
def test_elbo_loss(mocker, distribution, target, shape):
    distribution_object: distributions.Distribution = distribution(shape)
    target: VariationalLoss = target(shape).log_prob

    mocker.spy(distribution, 'sample') # track calls to distribution.sample

    loss = elbo_loss(
        distribution_object,
        target,
        key=random.PRNGKey(0),
    )

    assert loss.shape == () # expect scalar loss
    assert jnp.isfinite(loss) # expect finite loss

def test_variational_fit_e2e():
    # A simple E2E test to make sure that the function runs without error
    flow_random_key = random.PRNGKey(10)
    flow = MaskedAutoregressiveFlow(
        key=flow_random_key,
        base_dist=distributions.StandardNormal((5,)),
        flow_layers=4,
        transformer=bijections.RationalQuadraticSpline(knots=3, interval=6),
    )

    # Target a shifted + scaled normal
    target_dist = distributions.Normal(
        1.5 * jnp.ones(5), 
        0.5 * jnp.ones(5),
    )
    target = target_dist.log_prob

    train_random_key = random.PRNGKey(0)
    trained_flow, losses,  record = variational_fit(
        key=train_random_key,
        dist=flow,
        target=target,
        num_epochs=10,
        show_progress=False,
    )

    assert record is None

    # Check that we have trained the flow
    initial_params, initial_static = eqx.partition(flow, eqx.is_inexact_array)
    trained_params, trained_static = eqx.partition(trained_flow, eqx.is_inexact_array)
    
    assert initial_static == trained_static
    jax.tree_util.tree_map(
        lambda trained_param, initial_param: not jnp.allclose(trained_param, initial_param), 
        trained_params, 
        initial_params
    )

    # We expect the loss to be decreasing
    start, end = jnp.array(losses).split(2)
    assert jnp.mean(start) > jnp.mean(end)