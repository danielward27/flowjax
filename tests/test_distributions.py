import jax.numpy as jnp
import pytest
from jax import random

from flowjax.distributions import (
    Cauchy,
    Gumbel,
    Normal,
    StandardNormal,
    StudentT,
    Uniform,
    _StandardCauchy,
    _StandardGumbel,
    _StandardStudentT,
    _StandardUniform,
)

# This sets up a number of constructors shape -> instance for testing
# the generic API of Distribution classes.
# Note we do not test the private "Standard" distributions, assuming they are
# sufficiently tested by their loc, scale public counterparts.

_test_distributions = {
    "StandardNormal": StandardNormal,
    "Normal": lambda shape: Normal(jnp.zeros(shape)),
    "_StandardUniform": _StandardUniform,
    "Uniform": lambda shape: Uniform(jnp.zeros(shape), 1),
    "_StandardGumbel": _StandardGumbel,
    "Gumbel": lambda shape: Gumbel(jnp.zeros(shape)),
    "_StandardCauchy": _StandardCauchy,
    "Cauchy": lambda shape: Cauchy(jnp.zeros(shape)),
    "_StandardStudentT": lambda shape: _StandardStudentT(jnp.ones(shape)),
    "StudentT": lambda shape: StudentT(jnp.ones(shape)),
}



_test_distributions = [pytest.param(v, id=k) for k, v in _test_distributions.items()]
_test_shapes = [(), (2, ), (2,3)]

@pytest.mark.parametrize("distribution", _test_distributions)
@pytest.mark.parametrize("shape", _test_shapes)
def test_sample(distribution, shape):
    d = distribution(shape=shape)
    sample = d.sample(random.PRNGKey(0))
    assert sample.shape == shape

    sample_shape = (2,2)
    sample = d.sample(random.PRNGKey(0), sample_shape=sample_shape)
    assert sample.shape == sample_shape + shape


@pytest.mark.parametrize("distribution", _test_distributions)
@pytest.mark.parametrize("shape", _test_shapes)
def test_log_prob(distribution, shape):
    d = distribution(shape=shape)
    x = d.sample(random.PRNGKey(0))

    assert d.log_prob(x).shape == ()

    sample_shape = (2,3)
    x = d.sample(random.PRNGKey(0), sample_shape=sample_shape)
    assert d.log_prob(x).shape == sample_shape


# @pytest.mark.parametrize("distribution", _test_distributions)
# def test_sin_log_prob(distribution):
#     d = distribution(shape=3)
#     sample = d.sample(random.PRNGKey(0))
#     assert d.log_prob(sample).shape == ()


# @pytest.mark.parametrize("distribution_class", _test_distributions)
# def test_log_prob(distribution_class):
#     d = distribution_class(dim=3)
#     # sample 4 times from 3d distribution
#     x_matrix = d.sample(random.PRNGKey(0), n=4)
#     assert d.log_prob(x_matrix).shape == (4,)


@pytest.mark.parametrize("distribution", _test_distributions)
def test_log_prob_shape_mismatch(distribution):
    d = distribution(shape=(3,))

    with pytest.raises(ValueError):
        d.log_prob(jnp.ones((3,2)))

    d = distribution(shape=(3,2))
    with pytest.raises(ValueError):
        d.log_prob(jnp.ones((2,)))



def test_normal_params():
    dist = Normal(
        jnp.array([1.0, 2.0]),
        jnp.array([3.0, 4.0]),
    )

    assert dist.loc == pytest.approx(jnp.array([1.0, 2.0]))
    assert dist.scale == pytest.approx(jnp.array([3.0, 4.0]))


def test_uniform_params():
    dist = Uniform(
        jnp.array([1.0, 2.0]),
        jnp.array([3.0, 4.0]),
    )

    assert dist.minval == pytest.approx(jnp.array([1.0, 2.0]))
    assert dist.maxval == pytest.approx(jnp.array([3.0, 4.0]))
