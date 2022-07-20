# Distribution object (for flows and base distributions)

from abc import ABC, abstractmethod
from typing import Optional
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
import jax

# To construct a distribution, we define _log_prob and _sample, which take in vector arguments.
# More friendly methods are then created from these, supporting batches of inputs.
# Note that unconditional distributions should allow, but ignore the passing of conditional variables
# (to facilitate easy composing of conditional and unconditional distributions)


class Distribution(ABC):
    """Distribution base class."""

    dim: int
    cond_dim: int
    in_axes: tuple

    @abstractmethod
    def _log_prob(self, x: jnp.ndarray, condition: Optional[jnp.ndarray] = None):
        "Evaluate the log probability of point x."
        pass

    @abstractmethod
    def _sample(self, key: random.PRNGKey, condition: Optional[jnp.ndarray] = None):
        "Sample the distribution."
        pass

    @property
    def in_axes(self):
        "Controls whether to vmap over the conditioning variable."
        return (0, 0) if self.cond_dim > 0 else (0, None)

    @property
    def conditional(self):
        "Whether the distribution is an unconditional distribution or not."
        return True if self.cond_dim > 0 else False

    def sample(
        self,
        key: random.PRNGKey,
        condition: Optional[jnp.ndarray] = None,
        n: Optional[int] = None,
    ):
        self._argcheck(condition=condition)
        condition = jnp.empty((0,)) if condition is None else condition
        if n is None and condition.ndim == 1:  # No need to vmap in this case
            return self._sample(key, condition)
        else:
            in_axes, n = (
                ((0, 0), condition.shape[0]) if condition.ndim == 2 else ((0, None), n)
            )
            keys = random.split(key, n)
            return jax.vmap(self._sample, in_axes)(keys, condition)

    def log_prob(self, x: jnp.ndarray, condition: Optional[jnp.ndarray] = None):
        """Evaluate the log probability. If a matrix/matrices are passed,
        we vmap over the leading axis."""
        self._argcheck(x, condition)
        condition = jnp.empty((0,)) if condition is None else condition
        if x.ndim == 1 and condition.ndim == 1:  # No need to vmap in this case
            return self._log_prob(x, condition)
        else:
            in_axes = [0 if a.ndim == 2 else None for a in (x, condition)]
            return jax.vmap(self._log_prob, in_axes)(x, condition)

    def _argcheck(self, x=None, condition=None):
        if x is not None and x.shape[-1] != self.dim:
            raise ValueError(f"Expected x.shape[-1]=={self.dim}")
        if self.conditional and condition.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected condition.shape[-1]=={self.cond_dim}")


class Normal(Distribution):
    "Standard normal distribution, condition is ignored."

    def __init__(self, dim):
        self.dim = dim
        self.cond_dim = 0

    def _log_prob(self, x: jnp.ndarray, condition: Optional[jnp.ndarray] = None):
        assert x.shape == (self.dim,)
        return norm.logpdf(x).sum()

    def _sample(self, key: random.PRNGKey, condition: Optional[jnp.ndarray] = None):
        return random.normal(key, (self.dim,))
