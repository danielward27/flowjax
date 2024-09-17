from collections.abc import Callable
from math import prod
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.scipy.stats import multivariate_normal

from flowjax.bijections import AdditiveCondition, Affine, Exp
from flowjax.distributions import (
    AbstractDistribution,
    AbstractTransformed,
    Cauchy,
    Exponential,
    Gumbel,
    Laplace,
    Logistic,
    LogNormal,
    MultivariateNormal,
    Normal,
    StandardNormal,
    StudentT,
    Transformed,
    Uniform,
    VmapMixture,
    _StandardCauchy,
    _StandardGumbel,
    _StandardLaplace,
    _StandardLogistic,
    _StandardStudentT,
    _StandardUniform,
)

# This sets up a number of constructors shape -> instance for testing
# the generic API of Distribution classes.
# Note we do not test the private "Standard" distributions, assuming they are
# sufficiently tested by their loc, scale public counterparts.
_test_distributions = {
    # flowjax.distributions
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
    "LogNormal": lambda shape: LogNormal(jnp.ones(shape), 2),
    "_StandardLaplace": _StandardLaplace,
    "Laplace": lambda shape: Laplace(jnp.ones(shape)),
    "Exponential": lambda shape: Exponential(jnp.ones(shape)),
    "_StandardLogistic": _StandardLogistic,
    "Logistic": lambda shape: Logistic(jnp.ones(shape)),
    "VmapMixture": lambda shape: VmapMixture(
        eqx.filter_vmap(Normal)(jnp.arange(3 * prod(shape)).reshape(3, *shape)),
        weights=jnp.arange(3) + 1,
    ),
}


_test_distributions = [pytest.param(v, id=k) for k, v in _test_distributions.items()]
_test_shapes = [(), (2,), (2, 3)]


@pytest.mark.parametrize("distribution", _test_distributions)
@pytest.mark.parametrize("shape", _test_shapes)
def test_sample(distribution, shape):
    d = distribution(shape=shape)
    sample = d.sample(jr.key(0))
    assert sample.shape == shape

    sample_shape = (1, 2)
    sample = d.sample(jr.key(0), sample_shape)
    assert sample.shape == sample_shape + shape


@pytest.mark.parametrize("distribution", _test_distributions)
@pytest.mark.parametrize("shape", _test_shapes)
def test_log_prob(distribution, shape):
    d = distribution(shape=shape)
    x = d.sample(jr.key(0))

    assert d.log_prob(x).shape == ()

    sample_shape = (1, 2)
    x = d.sample(jr.key(0), sample_shape)
    assert d.log_prob(x).shape == sample_shape

    # test arraylike input
    assert jnp.all(d.log_prob(np.array(x)) == d.log_prob(x))

    if d.shape == ():
        assert d.log_prob(jnp.array(0)) == d.log_prob(0)


def test_uniform_params():
    dist = Uniform(
        jnp.array([1.0, 2.0]),
        jnp.array([3.0, 4.0]),
    )

    assert dist.minval == pytest.approx(jnp.array([1.0, 2.0]))
    assert dist.maxval == pytest.approx(jnp.array([3.0, 4.0]))


# Since the broadcasting behaviour is shared by all distributions
# we test it for a single unconditional and conditional distribution only.

dist_shape, sample_shape, condition_shape = ([(), (3, 4)] for _ in range(3))


class _TestDist(AbstractDistribution):
    "Toy distribution object, for testing of distribution broadcasting."
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None

    def __init__(self, shape, cond_shape=None):
        self.shape = shape
        self.cond_shape = cond_shape

    def _log_prob(self, x, condition=None):
        return jnp.zeros(())

    def _sample(self, key, condition=None):
        return jnp.zeros(self.shape)

    def _sample_and_log_prob(self, key, condition=None):
        return jnp.zeros(self.shape), jnp.zeros(())


def test_multivariate_normal():
    loc = jnp.arange(2)
    cov = jnp.array([[2, 0.5], [0.5, 3]])
    mvn = MultivariateNormal(loc, cov)

    key = jr.key(0)
    sample = mvn.sample(key)
    expected = pytest.approx(multivariate_normal.logpdf(sample, loc, cov))
    assert mvn.log_prob(sample) == expected
    assert mvn.covariance == pytest.approx(cov)


@pytest.mark.parametrize("dist_shape", dist_shape)
@pytest.mark.parametrize("sample_shape", sample_shape)
def test_broadcasting_unconditional(dist_shape, sample_shape):
    d = _TestDist(dist_shape)
    samples = d.sample(jr.key(0), sample_shape)
    assert samples.shape == sample_shape + dist_shape

    log_probs = d.log_prob(samples)
    assert log_probs.shape == sample_shape


@pytest.mark.parametrize("dist_shape", dist_shape)
@pytest.mark.parametrize("sample_shape", sample_shape)
@pytest.mark.parametrize("condition_shape", condition_shape)
@pytest.mark.parametrize(
    "leading_cond_shape",
    [(), (3, 4)],
)  # Additional leading dimensions in condition
def test_broadcasting_conditional(
    dist_shape,
    sample_shape,
    condition_shape,
    leading_cond_shape,
):
    key = jr.key(0)
    d = _TestDist(dist_shape, condition_shape)
    condition = jnp.zeros(leading_cond_shape + condition_shape)
    samples = d.sample(key, sample_shape, condition)
    assert samples.shape == sample_shape + leading_cond_shape + dist_shape

    log_probs = d.log_prob(samples, condition)
    assert log_probs.shape == sample_shape + leading_cond_shape

    samples, log_probs = d.sample_and_log_prob(key, sample_shape, condition)
    assert samples.shape == sample_shape + leading_cond_shape + dist_shape
    assert log_probs.shape == sample_shape + leading_cond_shape


test_cases = [
    StandardNormal(
        (2, 2),
    ),  # Won't have custom sample_and_log_prob implementation as not Transformed
    Normal(jnp.ones((2, 2))),  # Will have custom implementation as is Transformed
]


@pytest.mark.parametrize("dist", test_cases)
def test_sample_and_log_prob(dist):
    # We test broadcasting behaviour seperately above.
    # Just check consistency to seperately using methods
    key = jr.key(0)
    x_naive = dist._sample(key)
    lp_naive = dist._log_prob(x_naive)
    x, lp = dist._sample_and_log_prob(key)
    assert x == pytest.approx(x_naive)
    assert lp == pytest.approx(lp_naive)


def test_transformed_merge_transforms():
    shape = (3, 3)
    nested = Transformed(Normal(jnp.ones(shape)), Exp(shape))
    unnested = nested.merge_transforms()

    assert isinstance(nested.base_dist, AbstractTransformed)
    assert not isinstance(unnested.base_dist, AbstractTransformed)

    key = jr.key(0)
    sample = unnested.sample(key)
    assert pytest.approx(sample) == nested.sample(key)

    assert pytest.approx(nested.log_prob(sample)) == unnested.log_prob(sample)


class _TestCase(NamedTuple):
    name: str
    method: Callable
    args: tuple
    error: type
    match: str


test_cases = [
    _TestCase(
        name="missing condition sample",
        method=_TestDist((), ()).sample,
        args=(jr.key(0),),
        error=TypeError,
        match="Expected condition to be arraylike",
    ),
    _TestCase(
        name="wrong x shape log prob",
        method=_TestDist((2,), None).log_prob,
        args=(jnp.ones((3, 3)), None),
        error=ValueError,
        match="Expected trailing dimensions matching",
    ),
    _TestCase(
        name="wrong condition shape log prob",
        method=_TestDist((), (2, 3)).log_prob,
        args=(0, jnp.ones((3, 3))),
        error=ValueError,
        match="Expected trailing dimensions matching",
    ),
]


@pytest.mark.parametrize("test_case", test_cases, ids=[t.name for t in test_cases])
def test_method_errors(test_case):
    with pytest.raises(test_case.error, match=test_case.match):
        test_case.method(*test_case.args)


@pytest.mark.parametrize("weights", [jnp.array([0.25, 0.75]), jnp.array([1.0, 3])])
def test_vmap_mixture(weights):
    x = 3
    gaussian_mixture = VmapMixture(
        eqx.filter_vmap(Normal)(jnp.arange(2)),
        weights=weights,
    )
    normalized_weights = weights / weights.sum()
    expected = jnp.log(
        normalized_weights[0] * jnp.exp(Normal(0).log_prob(x))
        + normalized_weights[1] * jnp.exp(Normal(1).log_prob(x))
    )
    assert pytest.approx(expected) == gaussian_mixture.log_prob(x)


def test_transformed():
    # Unconditional base dist, conditional transform

    dist = Transformed(
        StandardNormal(),
        AdditiveCondition(lambda x: x.sum(), (), cond_shape=(2,)),
    )
    assert dist.sample(jr.key(0), condition=jnp.ones((5, 2))).shape == (5,)
    assert dist.shape == ()
    assert dist.cond_shape == (2,)

    # Conditional base_dist, unconditional transform
    dist = Transformed(dist, Affine())
    assert dist.sample(jr.key(0), condition=jnp.ones((5, 2))).shape == (5,)
    assert dist.shape == ()
    assert dist.cond_shape == (2,)
