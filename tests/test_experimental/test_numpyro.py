from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as ndist
import pytest
from equinox.nn import Linear
from jax.flatten_util import ravel_pytree
from numpyro import sample
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.optim import Adam

from flowjax.bijections import AdditiveCondition
from flowjax.distributions import Normal, StandardNormal, Transformed
from flowjax.experimental.numpyro import TransformedToNumpyro, register_params
from flowjax.flows import block_neural_autoregressive_flow

true_mean, true_std = jnp.ones(2), 2 * jnp.ones(2)


def numpyro_model():
    sample("x", TransformedToNumpyro(Normal(true_mean, true_std)))


def test_mcmc():
    "Check that flowjax distributions function in MCMC."
    key = jr.PRNGKey(0)
    mcmc = MCMC(NUTS(numpyro_model), num_warmup=50, num_samples=500)  # 2d N(1, 2)
    key, subkey = jr.split(key)
    mcmc.run(subkey)
    samps = mcmc.get_samples()["x"]
    assert pytest.approx(samps.mean(axis=0), abs=0.2) == true_mean
    assert pytest.approx(samps.std(axis=0), abs=0.2) == true_std

    def plate_model():
        # Note, in flowjax we do not have the concept of batch shape for simplicity.
        # We could add support in the numpyro context later if we wish. Note below,
        # the plate dim is -1 (as the flowjax normal has event dim (2,)).
        with numpyro.plate("obs", 10, dim=-1):
            sample("x", TransformedToNumpyro(Normal(true_mean, true_std)))

    mcmc = MCMC(NUTS(plate_model), num_warmup=50, num_samples=500)  # 2d N(1, 2)
    key, subkey = jr.split(key)
    mcmc.run(subkey)

    samps = mcmc.get_samples()["x"]
    assert samps.shape == (500, 10, 2)
    assert pytest.approx(samps.mean(axis=[0, 1]), abs=0.2) == true_mean
    assert pytest.approx(samps.std(axis=[0, 1]), abs=0.2) == true_std


def test_vi():
    "Check that flowjax distributions can be used as guide/variational distributions."

    def guide(dist):
        dist = register_params("guide", dist)
        dist = TransformedToNumpyro(dist)
        sample("x", dist)

    optimizer = Adam(step_size=0.01)

    guide_dist = Normal(jnp.zeros(2), 1)  # 2d N(0, 1)

    svi = SVI(numpyro_model, partial(guide, guide_dist), optimizer, loss=Trace_ELBO())
    svi_result = svi.run(jr.PRNGKey(0), num_steps=1000)

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
    svi_result = svi.run(jr.PRNGKey(0), num_steps=2)  # Check runs


def test_conditional_vi():
    # Similar to amortized variational inference
    dim = 2
    cond_dim = 3

    key, subkey = jr.split(jr.PRNGKey(0))

    true_dist, guide_dist = get_conditional_true_guide(subkey, dim, cond_dim)

    def model():
        cond = sample("cond", ndist.Normal(jnp.zeros((3,))))
        sample("x", TransformedToNumpyro(true_dist, cond))

    def guide(guide_dist):
        guide_dist = register_params("guide", guide_dist)
        cond = sample("cond", ndist.Normal(jnp.zeros((3,))))
        sample("x", TransformedToNumpyro(guide_dist, cond))

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
            sample("x", TransformedToNumpyro(Normal(true_mean, true_std)))

    guide_dist = Normal(jnp.ones_like(true_mean), jnp.ones_like(true_std))

    def guide(guide_dist):
        guide = register_params("guide", guide_dist)
        with numpyro.plate("obs", plate_dim):
            sample("x", TransformedToNumpyro(guide))

    guide = partial(guide, guide_dist)
    optimizer = Adam(step_size=0.01)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    svi_result = svi.run(jr.PRNGKey(0), num_steps=10000)

    guide_dist = eqx.combine(svi_result.params["guide"], guide_dist)

    assert pytest.approx(guide_dist.loc, abs=0.2) == true_mean
    assert pytest.approx(guide_dist.scale, abs=0.2) == true_std


key = jr.PRNGKey(0)
test_cases = [[(), ()], [(2,), ()], [(), (2,)], [(3, 2, 4), (1, 2)]]


@pytest.mark.parametrize(("shape", "sample_shape"), test_cases)
def test_TransformedToNumpyro(shape, sample_shape):
    key, subkey = jr.split(jr.PRNGKey(0))
    means = jr.normal(subkey, shape)

    key, subkey = jr.split(key)
    stds = jnp.exp(jr.normal(subkey, shape))

    dist = Normal(means, stds)
    wrapped = TransformedToNumpyro(dist)

    # Same key should lead to same samples
    x = dist.sample(key, sample_shape)
    assert wrapped.sample(key, sample_shape) == pytest.approx(x)

    # Should give same log prob
    assert wrapped.log_prob(x) == pytest.approx(dist.log_prob(x))


def test_batched_condition():
    dim = 2
    cond_dim = 3
    key, subkey = jr.split(jr.PRNGKey(0))
    dist = block_neural_autoregressive_flow(
        key,
        base_dist=Normal(jnp.zeros(dim)),
        cond_dim=cond_dim,
    )

    condition = jnp.ones((10, 3))
    wrapped = TransformedToNumpyro(dist, condition)
    assert wrapped.batch_shape == (10,)

    key, subkey = jr.split(key)
    true_dist, guide_dist = get_conditional_true_guide(subkey, dim, cond_dim)

    def model():
        with numpyro.plate("N", 10, dim=-2):
            cond = sample("cond", ndist.Normal(jnp.zeros(3)).to_event())
            sample("x", TransformedToNumpyro(true_dist, cond))

    def guide(guide_dist):
        guide_dist = register_params("guide", guide_dist)
        with numpyro.plate("N", 10, dim=-2):
            cond = sample("cond", ndist.Normal(jnp.zeros(3)).to_event())
            sample("x", TransformedToNumpyro(guide_dist, cond))

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
