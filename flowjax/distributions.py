# Distribution object (for flows and base distributions)

from abc import ABC, abstractmethod
from typing import Optional
import jax.numpy as jnp
from jax import random
from flowjax.utils import broadcast_except_last
from jax.scipy.stats import norm
import jax


# TODO We define _log_prob and _sample and the useful methods should be defined for us?


class Distribution(ABC):
    """Distribution base class (conditional or unconditional).
    x (and optionally condition) should be vectors."""
    dim: int
    cond_dim: int
    in_axes: tuple

    @abstractmethod
    def _log_prob(self, x: jnp.ndarray, condition: Optional[jnp.ndarray] = None):
        "Evaluate the log probability of point x."
        pass

    @abstractmethod
    def _sample(
        self, key: random.PRNGKey, condition: Optional[jnp.ndarray] = None,
    ):
        "Sample the distribution."
        pass

    def sample(
        self,
        key: random.PRNGKey,
        condition: Optional[jnp.ndarray] = None,
        n: Optional[int] = None
        ):
        if self.cond_dim > 0:
            n = condition.shape[0] if n is None else n
            condition = jnp.broadcast_to(condition, (n, self.cond_dim))

        keys = jnp.array(random.split(key, n))
        return jax.vmap(self._sample, self.in_axes)(keys, condition)
        
    def log_prob(self, x: jnp.ndarray, condition: Optional[jnp.ndarray] = None):
        x, condition = broadcast_except_last(x, condition)
        return jax.vmap(self._log_prob, self.in_axes)(x, condition)


class Normal(Distribution):
    "Standard normal, condition ignored."

    def __init__(self, dim):
        self.dim = dim
        self.cond_dim = 0
        self.in_axes = (0, 0) if self.cond_dim > 0 else (0, None)

    def _log_prob(self, x: jnp.ndarray, condition: Optional[jnp.ndarray] = None):
        assert x.shape == (self.dim,)
        return norm.logpdf(x).sum()

    def _sample(self, key: random.PRNGKey, condition: Optional[jnp.ndarray] = None):
        return random.normal(key, (self.dim, ))

    
# class BatchedDistribution:
#     """Distribution which supports batching of sampling and log_prob,
#     with sensible broadcasting."""
#     dim: int
#     cond_dim: int
#     in_axes: tuple

#     def __init__(self, distribution: Distribution):
#         self.distribution = distribution
#         self.dim = distribution.dim
#         self.cond_dim = distribution.cond_dim
#         self.in_axes = (0, 0) if self.cond_dim > 0 else (0, None)

#     def sample(
#         self,
#         key: random.PRNGKey,
#         condition: Optional[jnp.ndarray] = None,
#         n: Optional[int] = None
#         ):
#         if self.cond_dim > 0:
#             n = condition.shape[0] if n is None else n
#             condition = jnp.broadcast_to(condition, (n, self.cond_dim))

#         keys = jnp.array(random.split(key, n))
#         return jax.vmap(self.distribution.sample, self.in_axes)(keys, condition)
        
#     def log_prob(self, x: jnp.ndarray, condition: Optional[jnp.ndarray] = None):
#         x, condition = broadcast_except_last(x, condition)
#         return jax.vmap(self.distribution.log_prob, self.in_axes)(x, condition)


