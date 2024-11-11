from functools import partial
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as ndist
import pytest
from equinox.nn import Linear
from jax.flatten_util import ravel_pytree
from jaxtyping import Array
from numpyro import handlers
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.optim import Adam

from flowjax.bijections import AbstractBijection, AdditiveCondition, Affine, Invert
from flowjax.distributions import (
    LogNormal,
    Normal,
    StandardNormal,
    Transformed,
)
from flowjax.experimental.numpyro import (
    _BijectionToNumpyro,
    _get_batch_shape,
    distribution_to_numpyro,
    register_params,
    sample,
)
from flowjax.flows import block_neural_autoregressive_flow, masked_autoregressive_flow

true_mean, true_std = jnp.ones(2), 2 * jnp.ones(2)


def numpyro_model():
    sample("x", Normal(true_mean, true_std))


def test_get_batch_shape():
    assert _get_batch_shape(jnp.ones((2, 3)), ()) == (2, 3)
    assert _get_batch_shape(jnp.ones(3), (3,)) == ()
    assert _get_batch_shape(jnp.ones((1, 2, 3)), (2, 3)) == (1,)


def test_mcmc():
    "Check that flowjax distributions function in MCMC."
    key = jr.key(0)
    mcmc = MCMC(NUTS(numpyro_model), num_warmup=50, num_samples=500)  # 2d N(1, 2)
    key, subkey = jr.split(key)
    mcmc.run(subkey)
    samps = mcmc.get_samples()["x"]
    assert pytest.approx(samps.mean(axis=0), abs=0.2) == true_mean
    assert pytest.approx(samps.std(axis=0), abs=0.2) == true_std

    def plate_model():
        # Note, in FlowJAX we do not have the concept of batch shape for simplicity.
        # We could add support in the numpyro context later if we wish. Note below,
        # the plate dim is -1 (as the FlowJAX normal has event dim (2,)).
        with numpyro.plate("obs", 10, dim=-1):
            sample("x", Normal(true_mean, true_std))

    mcmc = MCMC(NUTS(plate_model), num_warmup=50, num_samples=500)  # 2d N(1, 2)
    key, subkey = jr.split(key)
    mcmc.run(subkey)

    samps = mcmc.get_samples()["x"]
    assert samps.shape == (500, 10, 2)
    assert pytest.approx(samps.mean(axis=[0, 1]), abs=0.2) == true_mean
    assert pytest.approx(samps.std(axis=[0, 1]), abs=0.2) == true_std


def test_vi():
    "Check that FlowJAX distributions can be used as guide/variational distributions."

    def guide(dist):
        dist = register_params("guide", dist)
        sample("x", dist)

    optimizer = Adam(step_size=0.01)

    guide_dist = Normal(jnp.zeros(2), 1)  # 2d N(0, 1)

    svi = SVI(numpyro_model, partial(guide, guide_dist), optimizer, loss=Trace_ELBO())
    svi_result = svi.run(jr.key(0), num_steps=1000)

    guide_dist = eqx.combine(svi_result.params["guide"], guide_dist)

    assert pytest.approx(guide_dist.loc, abs=0.2) == true_mean
    assert pytest.approx(guide_dist.scale, abs=0.2) == true_std

    # Test intermediates are used - note BNAF has no inverse so intermediates
    # are required to compute the log_prob in VI.
    guide_dist = block_neural_autoregressive_flow(
        key,
        base_dist=Normal(jnp.zeros(2)),
        invert=False,
        nn_block_dim=1,
    )
    svi = SVI(numpyro_model, partial(guide, guide_dist), optimizer, loss=Trace_ELBO())
    svi_result = svi.run(jr.key(0), num_steps=2)  # Check runs


def test_conditional_vi():
    # Similar to amortized variational inference
    dim = 2
    cond_dim = 3

    key, subkey = jr.split(jr.key(0))

    true_dist, guide_dist = get_conditional_true_guide(subkey, dim, cond_dim)

    def model():
        cond = sample("cond", ndist.Normal(jnp.zeros((3,))))
        sample("x", true_dist, condition=cond)

    def guide(guide_dist):
        guide_dist = register_params("guide", guide_dist)
        cond = sample("cond", ndist.Normal(jnp.zeros((3,))))
        sample("x", guide_dist, condition=cond)

    optimizer = Adam(step_size=0.01)
    svi = SVI(model, partial(guide, guide_dist), optimizer, loss=Trace_ELBO())

    key, subkey = jr.split(key)
    svi_result = svi.run(subkey, num_steps=1000)

    trained_guide = eqx.combine(svi_result.params["guide"], guide_dist)
    true_params = ravel_pytree(true_dist)[0]
    norm_from_true_init = jnp.linalg.norm(true_params - ravel_pytree(guide_dist)[0])
    norm_from_true_final = jnp.linalg.norm(true_params - ravel_pytree(trained_guide)[0])

    # Arbitrarily, we check the l2 norm between the trained parameters and the truth
    # has more than halved over the course of training
    assert norm_from_true_final < 0.5 * norm_from_true_init


def test_vi_plate():
    plate_dim = 10

    def model():
        with numpyro.plate("obs", plate_dim):
            sample("x", Normal(true_mean, true_std))

    guide_dist = Normal(jnp.ones_like(true_mean), jnp.ones_like(true_std))

    def guide(guide_dist):
        guide = register_params("guide", guide_dist)
        with numpyro.plate("obs", plate_dim):
            sample("x", guide)

    guide = partial(guide, guide_dist)
    optimizer = Adam(step_size=0.01)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    svi_result = svi.run(jr.key(0), num_steps=10000)

    guide_dist = eqx.combine(svi_result.params["guide"], guide_dist)

    assert pytest.approx(guide_dist.loc, abs=0.2) == true_mean
    assert pytest.approx(guide_dist.scale, abs=0.2) == true_std


key = jr.key(0)
test_cases = {
    "distribution": [
        StandardNormal(()),
        StandardNormal((1, 2)),
        Normal(),
        Normal(jnp.ones((1, 2))),
    ],
    "sample_shape": [(), (3, 4)],
}


@pytest.mark.parametrize("distribution", test_cases["distribution"])
@pytest.mark.parametrize("sample_shape", test_cases["sample_shape"])
def test_distribution_to_numpyro(distribution, sample_shape):
    wrapped = distribution_to_numpyro(distribution)

    # Same key should lead to same samples
    x = distribution.sample(key, sample_shape)
    assert wrapped.sample(key, sample_shape) == pytest.approx(x)

    # Should give same log prob
    assert wrapped.log_prob(x) == pytest.approx(distribution.log_prob(x))


def test_batched_condition():
    dim = 2
    cond_dim = 3
    key, subkey = jr.split(jr.key(0))
    dist = block_neural_autoregressive_flow(
        key,
        base_dist=Normal(jnp.zeros(dim)),
        cond_dim=cond_dim,
    )

    condition = jnp.ones((10, 3))
    wrapped = distribution_to_numpyro(dist, condition)
    assert wrapped.batch_shape == (10,)

    key, subkey = jr.split(key)
    true_dist, guide_dist = get_conditional_true_guide(subkey, dim, cond_dim)

    def model():
        with numpyro.plate("N", 10, dim=-2):
            cond = sample("cond", ndist.Normal(jnp.zeros(3)).to_event())
            sample("x", true_dist, condition=cond)

    def guide(guide_dist):
        guide_dist = register_params("guide", guide_dist)
        with numpyro.plate("N", 10, dim=-2):
            cond = sample("cond", ndist.Normal(jnp.zeros(3)).to_event())
            sample("x", guide_dist, condition=cond)

    optimizer = Adam(step_size=0.01)
    svi = SVI(model, partial(guide, guide_dist), optimizer, loss=Trace_ELBO())
    key, subkey = jr.split(key)
    svi_result = svi.run(subkey, num_steps=1000)

    trained_guide = eqx.combine(svi_result.params["guide"], guide_dist)
    true_params = ravel_pytree(true_dist)[0]
    norm_from_true_init = jnp.linalg.norm(true_params - ravel_pytree(guide_dist)[0])
    norm_from_true_final = jnp.linalg.norm(true_params - ravel_pytree(trained_guide)[0])

    # Arbitrarily, we check the l2 norm between the trained parameters and the truth
    # has more than halved over the course of training
    assert norm_from_true_final < 0.5 * norm_from_true_init


def get_conditional_true_guide(key, dim, cond_dim):
    true_dist, guide_dist = (
        Transformed(
            StandardNormal((dim,)),
            AdditiveCondition(Linear(cond_dim, dim, key=k), (dim,), (cond_dim,)),
        )
        for k in jr.split(key)
    )
    return true_dist, guide_dist


def test_callable_params():
    class MyParams(eqx.Module):
        a: Array
        b: Array

        def __call__(self, *args, **kwargs):
            raise NotImplementedError()

    def model(params):
        params = register_params("my_params", params)
        sample("x", Normal(params.a, params.b))

    my_params = MyParams(jnp.ones(3), jnp.ones(3))
    model = handlers.seed(model, key)
    model_trace = handlers.trace(model).get_trace(my_params)

    assert "my_params" in model_trace
    assert eqx.tree_equal(my_params, model_trace["my_params"]["value"])


def test_expected_elbo():
    dim = 2
    cond_dim = 3
    n_obs = 5
    key, subkey = jr.split(jr.key(0))

    likelihood = block_neural_autoregressive_flow(
        key,
        base_dist=Normal(jnp.zeros(dim)),
        cond_dim=cond_dim,
        nn_block_dim=2,
    )
    prior = StandardNormal((cond_dim,))

    key, subkey = jr.split(jr.key(0))
    posterior = block_neural_autoregressive_flow(
        key,
        base_dist=Normal(jnp.zeros(cond_dim)),
        cond_dim=dim,
        nn_block_dim=2,
        invert=False,
    )

    def model(obs):
        with numpyro.plate("obs", n_obs):
            theta = sample("theta", prior)
            sample("x", likelihood, obs=obs, condition=theta)

    def guide(obs):
        with numpyro.plate("obs", n_obs):
            sample("theta", posterior, condition=obs)

    def manual_elbo(key):
        guide_samps, guide_lps = posterior.sample_and_log_prob(key, condition=obs)
        model_lps = prior.log_prob(guide_samps) + likelihood.log_prob(obs, guide_samps)
        return model_lps.sum() - guide_lps.sum()

    from numpyro.infer import Trace_ELBO

    key, subkey = jr.split(key)
    obs = jr.normal(subkey, (n_obs, dim))

    key, subkey = jr.split(key)
    elbo = -Trace_ELBO(num_particles=500).loss(subkey, {}, model, guide, obs)

    # Seed handling is different, so we compare stochastic elbo for simplicity
    manual = jax.vmap(manual_elbo)(jr.split(subkey, 500)).mean()
    assert elbo == pytest.approx(manual, abs=1)


def test_conditional_sample_and_log_prob_with_intermediates():

    cond_flow = masked_autoregressive_flow(
        jr.key(0),
        base_dist=Normal(jnp.ones(3)),
        cond_dim=2,
        flow_layers=2,
    )
    condition = jnp.linspace(-1, 1, num=20).reshape((10, 2))
    dist = distribution_to_numpyro(cond_flow, condition=condition)

    assert dist.batch_shape == (condition.shape[0],)
    assert dist.base_dist.batch_shape == (condition.shape[0],)

    sample, intermediates = dist.sample_with_intermediates(key)
    z, log_det = intermediates[0]
    assert sample.shape == z.shape
    assert log_det.shape == (condition.shape[0],)
    assert not pytest.approx(sample[0]) == sample[1]  # Ensures different base sample

    expected = cond_flow.log_prob(sample, condition)
    log_probs = dist.log_prob(sample, intermediates)
    assert pytest.approx(expected) == log_probs


def test_BijectionToNumpyro_unconditional():
    loc, scale = jnp.arange(3), jnp.arange(1, 4)
    affine = _BijectionToNumpyro(Affine(loc, scale))
    npro_affine = AffineTransform(
        loc,
        scale,
    )

    x = jnp.array([2, 1, 3])
    assert pytest.approx(affine(x)) == npro_affine(x)
    assert pytest.approx(affine.inv(x)) == npro_affine.inv(x)
    assert (
        pytest.approx(affine.log_abs_det_jacobian(x, y=None))
        == npro_affine.log_abs_det_jacobian(x, y=None).sum()
    )

    transformed, npro_transformed = (
        numpyro.distributions.TransformedDistribution(
            numpyro.distributions.Normal(jnp.zeros(3)),
            aff,
        )
        for aff in [affine, npro_affine]
    )
    expected = npro_transformed.log_prob(jnp.arange(9).reshape(3, 3)).sum(axis=-1)
    observed = transformed.log_prob(jnp.arange(9).reshape(3, 3))
    assert pytest.approx(expected) == observed


def test_invalid_domains_BijectionToNumpyro():
    affine = Affine(jnp.arange(3), jnp.arange(1, 4))

    with pytest.raises(ValueError, match="domain.event_dim"):
        _BijectionToNumpyro(affine, domain=numpyro.distributions.constraints.real)

    with pytest.raises(ValueError, match="codomain.event_dim"):
        _BijectionToNumpyro(affine, codomain=numpyro.distributions.constraints.real)


def test_transformed_reparam():
    loc, scale = 2, 3

    config = {"x": numpyro.infer.reparam.TransformReparam()}
    log_norm = LogNormal(loc, scale)

    def model():
        with numpyro.handlers.reparam(config=config):
            sample("x", log_norm)

    trace = handlers.trace(
        handlers.seed(model, jr.key(0)),
    ).get_trace()

    assert "x_base" in trace.keys()
    expected_x = log_norm.merge_transforms().bijection.transform(
        trace["x_base"]["value"]
    )
    assert pytest.approx(expected_x) == trace["x"]["value"]


class _ForwardOnly(AbstractBijection):
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def transform_and_log_det(self, x, condition=None):
        return x, jnp.zeros(())

    def inverse_and_log_det(self, y, condition=None):
        raise NotImplementedError()


def test_sampling_forward_only():
    dist = Transformed(
        StandardNormal(),
        _ForwardOnly(),
    )
    dist = distribution_to_numpyro(dist)
    dist.sample(jr.key(0))


def test_log_prob_inverse_only():
    dist = Transformed(
        StandardNormal(),
        Invert(_ForwardOnly()),
    )
    dist = distribution_to_numpyro(dist)
    dist.log_prob(0)
