# Distribution object (for flows and base distributions)

from abc import ABC, abstractmethod
from typing import Optional
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
import jax
from jax.random import KeyArray
from flowjax.utils import Array
from typing import Any

# To construct a distribution, we define _log_prob and _sample, which take in vector arguments.
# More friendly methods are then created from these, supporting batches of inputs.
# Note that unconditional distributions should allow, but ignore the passing of conditional variables
# (to facilitate easy composing of conditional and unconditional distributions)


class Distribution(ABC):
    """Distribution base class."""

    dim: int
    cond_dim: int

    @abstractmethod
    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        "Evaluate the log probability of point x."
        pass

    @abstractmethod
    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
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
        self, key: KeyArray, condition: Optional[Array] = None, n: Optional[int] = None,
    ) -> Array:
        """Sample from a distribution. The condition can be a vector, or a matrix.
        - If condition.ndim==1, n allows repeated sampling (a single sample is drawn if n is not provided).
        - If condition.ndim==2, axis 0 is treated as batch dimension, (one sample is drawn for each row).

        Args:
            key (KeyArray): Jax PRNGKey.
            condition (Optional[Array], optional): _description_. Defaults to None.
            n (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Array: Jax array of samples.
        """
        self._argcheck(condition=condition)

        if condition is None:
            condition = jnp.empty((0,))

        if n is None and condition.ndim == 1:  # No need to vmap in this case
            return self._sample(key, condition)
        else:
            if condition.ndim == 2:
                assert n is None, "n should not be provided when condition.ndim==2."
                in_axes = (0, 0)  # type: tuple[Any, Any]
                n = condition.shape[0]
            else:
                assert n is not None, "n should be provided when condition.ndim==1."
                in_axes = (0, None)

            keys = random.split(key, n)
            return jax.vmap(self._sample, in_axes)(keys, condition)

    def log_prob(self, x: Array, condition: Optional[Array] = None):
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
        if self.conditional:
            assert condition is not None
            if condition.shape[-1] != self.cond_dim:
                raise ValueError(f"Expected condition.shape[-1]=={self.cond_dim}")
            if condition.ndim > 2:
                raise ValueError("Expected condition.ndim to be 1 or 2.")


class Normal(Distribution):
    "Standard normal distribution, condition is ignored."

    def __init__(self, dim):
        self.dim = dim
        self.cond_dim = 0

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        assert x.shape == (self.dim,)
        return norm.logpdf(x).sum()

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.normal(key, (self.dim,))
