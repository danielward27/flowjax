# Distribution object (for flows and base distributions)

from abc import ABC, abstractmethod
from typing import Optional
from flowjax.bijections.abc import Bijection
from flowjax.bijections.affine import Affine
from jax import random
from jax.scipy import stats as jstats
import jax
import jax.numpy as jnp
from jax.random import KeyArray
from flowjax.utils import Array
from typing import Any
import equinox as eqx

# To construct a distribution, we define _log_prob and _sample, which take in vector arguments.
# More friendly methods are then created from these, supporting batches of inputs.
# Note that unconditional distributions should allow, but ignore the passing of conditional variables
# (to facilitate easy composing of conditional and unconditional distributions)


class Distribution(eqx.Module, ABC):
    """Distribution base class."""

    dim: int
    cond_dim: int

    @abstractmethod
    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        "Evaluate the log probability of point x."
        pass

    @abstractmethod
    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        "Sample a point from the distribution."
        pass

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
            condition (Optional[Array], optional): Conditioning variables. Defaults to None.
            n (Optional[int], optional): Number of samples (if condition.ndim==1). Defaults to None.

        Returns:
            Array: Jax array of samples.
        """
        self._argcheck_condition(condition)

        if n is None:
            if condition is None:
                return self._sample(key)
            elif condition.ndim == 1:
                return self._sample(key, condition)
            else:
                n = condition.shape[0]
                in_axes = (0, 0)  # type: tuple[Any, Any]
        else:
            if condition is not None and condition.ndim != 1:
                raise ValueError("condition must be 1d if n is provided.")
            in_axes = (0, None)
            
        keys = random.split(key, n)
        return jax.vmap(self._sample, in_axes)(keys, condition)

    def log_prob(self, x: Array, condition: Optional[Array] = None):
        """Evaluate the log probability. If a matrix/matrices are passed,
        we vmap (vectorise) over the leading axis.

        Args:
            x (Array): Points at which to evaluate density.
            condition (Optional[Array], optional): Conditioning variables. Defaults to None.

        Returns:
            Array: Jax array of log probabilities.
        """
        self._argcheck_x(x)
        self._argcheck_condition(condition)

        if condition is None:
            if x.ndim == 1:
                return self._log_prob(x)
            else:
                return jax.vmap(self._log_prob)(x)
        else:
            if (x.ndim == 1) and (condition.ndim == 1):
                return self._log_prob(x, condition)
            else:
                in_axes = [0 if a.ndim == 2 else None for a in (x, condition)]
                return jax.vmap(self._log_prob, in_axes)(x, condition)

    def _argcheck_x(self, x: Array):
        if x.ndim not in (1,2):
            raise ValueError("x.ndim should be 1 or 2")

        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected x.shape[-1]=={self.dim}, got {x.shape}.")

    def _argcheck_condition(self, condition: Optional[Array] = None):
        if condition is None:
            if self.conditional:
                raise ValueError(f"condition must be provided.")
        else:
            if condition.ndim not in (1,2):
                raise ValueError("condition.ndim should be 1 or 2")
            if condition.shape[-1] != self.cond_dim:
                raise ValueError(f"Expected condition.shape[-1]=={self.cond_dim}.")


class Transformed(Distribution):
    base_dist: Distribution
    bijection: Bijection
    dim: int
    cond_dim: int

    def __init__(
        self,
        base_dist: Distribution,
        bijection: Bijection,
    ):
        """Form a distribution like object using a base distribution and a
        bijection. We take the forward bijection for use in sampling, and the inverse
        bijection for use in density evaluation.

        Args:
            base_dist (Distribution): Base distribution.
            bijection (Bijection): Bijection defined in "normalising" direction.
        """
        self.base_dist = base_dist
        self.bijection = bijection
        self.dim = self.base_dist.dim
        self.cond_dim = max(self.bijection.cond_dim, self.base_dist.cond_dim)

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        z, log_abs_det = self.bijection.inverse_and_log_abs_det_jacobian(x, condition)
        p_z = self.base_dist._log_prob(z, condition)
        return p_z + log_abs_det

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        z = self.base_dist._sample(key, condition)
        x = self.bijection.transform(z, condition)
        return x


class StandardNormal(Distribution):
    """
    Implements a standard normal distribution, condition is ignored.
    """
    def __init__(self, dim: int):
        """
        Args:
            dim (int): Dimension of the normal distribution.
        """
        self.dim = dim
        self.cond_dim = 0

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.norm.logpdf(x).sum()

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.normal(key, (self.dim,))


class Normal(Transformed):
    """
    Implements an independent Normal distribution with mean and std for
    each dimension.
    """
    def __init__(self, mean: Array, std: Array):
        """
        Args:
            mean (Array): Array of the means of each dimension
            std (Array): Array of the standard deviations of each dimension
        """
        dim = mean.shape[0]

        super(Normal, self).__init__(
            base_dist=StandardNormal(dim),
            bijection=Affine(loc=mean, scale=std)
        )

    @property
    def mean(self):
        return self.bijection.loc

    @property
    def std(self):
        return self.bijection.scale

class StandardUniform(Distribution):
    """
    Implements a standard independent Uniform distribution, ie X ~ Uniform([0, 1]^dim).
    """
    def __init__(self, dim):
        self.dim = dim
        self.cond_dim = 0

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.uniform.logpdf(x).sum()

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.uniform(key, shape=(self.dim,))


class Uniform(Transformed):
    """
    Implements an independent uniform distribution 
    between min and max for each dimension.
    """
    def __init__(self, min: Array, max: Array):
        """
        Args:
            min (Array): ith entry gives the min of the ith dimension
            max (Array): ith entry gives the max of the ith dimension
        """
        if jnp.any(max < min):
            raise ValueError("Minimums must be less than maximums.")
        dim = min.shape[0]

        super(Uniform, self).__init__(
            base_dist=StandardUniform(dim),
            bijection=Affine(loc=min, scale=max - min)
        )

    @property
    def min(self):
        return self.bijection.loc

    @property
    def max(self):
        return self.bijection.loc + self.bijection.scale


class Gumbel(Distribution):
    """
    Implements standard gumbel distribution (loc=0, scale=1)
    Ref: https://en.wikipedia.org/wiki/Gumbel_distribution
    """
    def __init__(self, dim):
        self.dim = dim
        self.cond_dim = 0

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return -(x + jnp.exp(-x)).sum()

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.gumbel(key, shape=(self.dim,))


class Cauchy(Distribution):
    """
    Implements standard cauchy distribution (loc=0, scale=1)
    Ref: https://en.wikipedia.org/wiki/Cauchy_distribution
    """
    def __init__(self, dim):
        self.dim = dim
        self.cond_dim = 0

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.cauchy.logpdf(x).sum()

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.cauchy(key, shape=(self.dim,))


class StudentT(Distribution):
    """
    Implements student T distribution with specified degree of freedom.
    """
    dfs: Array
    def __init__(self, dim, dfs: Array):
        self.dim = dim
        self.cond_dim = 0
        self.df = dfs

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.t.logpdf(x, df=self.dfs).sum()

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.t(key, df=self.dfs, shape=(self.dim,))