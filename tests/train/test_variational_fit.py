import pytest

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

from flowjax.train.variational_fit import (
    elbo_loss, 
    fit_to_variational_target,
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
def test_elbo_loss(distribution, target, shape):
    distribution_object: distributions.Distribution = distribution(shape)
    target = target(shape).log_prob

    loss = elbo_loss(
        random.PRNGKey(0),
        distribution_object,
        target,
        num_samples=100
    )
    
    assert loss.shape == () # expect scalar loss
    assert jnp.isfinite(loss) # expect finite loss

def test_fit_to_variational_target_e2e():
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
    trained_flow, losses = fit_to_variational_target(
        key=train_random_key,
        dist=flow,
        target=target,
        steps=10,
        show_progress=False,
    )

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