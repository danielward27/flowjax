
import pytest
import jax.numpy as jnp
from jax import random
from flowjax.distributions import (
    StandardNormal,
    Uniform,
    Gumbel,
    Cauchy,
    StudentT,
)

_test_distributions = [
    pytest.param(StandardNormal, id='StandardNormal'),
    pytest.param(Uniform, id='Uniform'),
    pytest.param(Gumbel, id='Gumbel'),
    pytest.param(Cauchy, id='Cauchy'),
    pytest.param(StudentT, id='StudentT'),
]

@pytest.mark.parametrize('distribution_class', _test_distributions)
@pytest.mark.parametrize('dimension', [1, 3])
def test_single_sample(distribution_class, dimension):
    d = distribution_class(dim=dimension)
    sample = d.sample(random.PRNGKey(0))
    assert sample.shape == (dimension,)

@pytest.mark.parametrize('distribution_class', _test_distributions)
@pytest.mark.parametrize('dimension', [1, 3])
def test_sample(distribution_class, dimension):    
    d = distribution_class(dim=dimension)
    # sample 4 times from distribution
    x_matrix = d.sample(random.PRNGKey(0), n=4)

    assert jnp.any(x_matrix[0] != x_matrix[1])
    assert x_matrix.shape == (4, dimension)

@pytest.mark.parametrize('distribution_class', _test_distributions)
def test_log_prob(distribution_class):
    d = distribution_class(dim=3)
    x_matrix = d.sample(random.PRNGKey(0), n=4)
    assert d.log_prob(x_matrix).shape == (4,)

@pytest.mark.parametrize('distribution_class', _test_distributions)
def test_single_log_prob(distribution_class):
    d = distribution_class(dim=3)
    sample = d.sample(random.PRNGKey(0))
    assert d.log_prob(sample).shape == ()

@pytest.mark.parametrize('distribution_class', _test_distributions)
def test_log_prob(distribution_class):
    d = distribution_class(dim=3)
    # sample 4 times from 3d distribution
    x_matrix = d.sample(random.PRNGKey(0), n=4)
    assert d.log_prob(x_matrix).shape == (4,)

@pytest.mark.parametrize('distribution_class', _test_distributions)
@pytest.mark.parametrize('input_dimension', [1, 4])
def test_dimension_mismatch(distribution_class, input_dimension):
    d = distribution_class(dim=3)
    with pytest.raises(ValueError):
        d.log_prob(jnp.ones(input_dimension))

    with pytest.raises(ValueError):
        d.log_prob(jnp.ones((2, input_dimension)))