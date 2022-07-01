
import pytest
from flowjax.distributions import Normal
import jax.numpy as jnp
from jax import random

def test_Normal():
    d = Normal(dim=3)

    x_vec = d.sample(random.PRNGKey(0))
    assert x_vec.shape == (3,)
    
    x_matrix = d.sample(random.PRNGKey(0), n=4)
    assert jnp.any(x_matrix[0] != x_matrix[1])
    assert x_matrix.shape == (4,3)
    assert d.log_prob(x_matrix).shape == (4,)

    x_vec = jnp.ones(3)
    assert d.log_prob(x_vec).shape == ()

    with pytest.raises(ValueError):
        d.log_prob(jnp.ones(4))

    with pytest.raises(ValueError):
        d.log_prob(jnp.ones((4,4)))

