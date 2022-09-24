import pytest
import jax.numpy as jnp
from jax import random
from flowjax.distributions import (
    StandardNormal,
    Normal,
    StandardUniform,
    Uniform,
    StandardGumbel,
    Gumbel,
    StandardCauchy,
    Cauchy,
    StandardStudentT,
    StudentT,
)

# This sets up a number of constructors dim -> instance for testing 
# the generic API of Distribution classes.


_test_distributions = [
    (StandardNormal, 'StandardNormal'),
    (lambda dim: Normal(jnp.zeros(dim), jnp.ones(dim)), 'Normal'),
    (StandardUniform, 'StandardUniform'),
    (lambda dim: Uniform(jnp.zeros(dim), jnp.ones(dim)), 'Uniform'),
    (StandardGumbel, 'StandardGumbel'),
    (lambda dim: Gumbel(jnp.zeros(dim), jnp.ones(dim)), 'Gumbel'),
    (StandardCauchy, 'StandardCauchy'),
    (lambda dim: Cauchy(jnp.zeros(dim), jnp.ones(dim)), 'Cauchy'),
    (lambda dim: StandardStudentT(jnp.ones(dim)), "StandardStudentT"),
    (lambda dim: StudentT(jnp.ones(dim), jnp.zeros(dim), jnp.ones(dim)), 'StudentT')
]

_test_distributions = [pytest.param(a[0], id=a[1]) for a in _test_distributions]


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

def test_normal_params():
    dist = Normal(
        jnp.array([1., 2.]),
        jnp.array([3., 4.]),
    )

    assert dist.loc == pytest.approx(jnp.array([1., 2.]))
    assert dist.scale == pytest.approx(jnp.array([3., 4.]))

def test_uniform_params():
    dist = Uniform(
        jnp.array([1., 2.]),
        jnp.array([3., 4.]),
    )

    assert dist.minval == pytest.approx(jnp.array([1., 2.]))
    assert dist.maxval == pytest.approx(jnp.array([3., 4.]))