import pytest

import jax.numpy as jnp
from jax import random

from flowjax.train.variational_fit import (
    elbo_loss, 
    variational_fit,
    VariationalLoss, 
    VariationalTarget
)
from flowjax import distributions

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