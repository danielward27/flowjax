# %%
import pytest
from flowjax.distributions import Normal
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import random


def test_Normal():
    d = Normal(dim=3)
    x = d.sample(random.PRNGKey(0), n=4)

    assert jnp.any(x[0] != x[1])
    assert x.shape == (4,3)
    assert d.log_prob(x=x).shape == (4,)


# def test_Normal():
#     d = Normal(dim=3)

#     with pytest.raises(AssertionError):
#         d.log_prob(jnp.ones(5))

#     with pytest.raises(AssertionError):
#         d.log_prob(jnp.ones(5))

#     x = d.log_prob(jnp.zeros(3))
#     assert x.shape == ()

