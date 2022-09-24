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
    def __init__(self, loc: Array, scale: Array):
        """
        Args:
            loc (Array): Array of the means of each dimension.
            scale (Array): Array of the standard deviations of each dimension.
        """
        base_dist = StandardNormal(loc.shape[0])
        bijection = Affine(loc=loc, scale=scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def scale(self):
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
    def __init__(self, minval: Array, maxval: Array):
        """
        Args:
            minvals (Array): ith entry gives the min of the ith dimension
            maxvals (Array): ith entry gives the max of the ith dimension
        """
        if jnp.any(maxval < minval):
            raise ValueError("Minimums must be less than maximums.")
        base_dist = StandardUniform(minval.shape[0])
        bijection = Affine(loc=minval, scale=maxval - minval)
        super().__init__(base_dist, bijection)

    @property
    def minval(self):
        return self.bijection.loc

    @property
    def maxval(self):
        return self.bijection.loc + self.bijection.scale


class StandardGumbel(Distribution):
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


class Gumbel(Transformed):
    def __init__(self, loc: Array, scale: Array):
        base_dist = StandardGumbel(loc.shape[0])
        bijection = Affine(loc, scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def scale(self):
        return self.bijection.scale

    
class StandardCauchy(Distribution):
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


class Cauchy(Transformed):
    "Cauchy distribution with loc and scale."
    def __init__(self, loc: Array, scale: Array):
        base_dist = StandardGumbel(loc.shape[0])
        bijection = Affine(loc, scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def scale(self):
        return self.bijection.scale


class StandardStudentT(Distribution):
    """
    Implements student T distribution with specified degrees of freedom.
    """
    log_df: Array

    def __init__(self, df: Array):
        self.dim = df.shape[0]
        self.cond_dim = 0
        self.log_df = jnp.log(df)

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.t.logpdf(x, df=self.df).sum()

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.t(key, df=self.df, shape=(self.dim,))

    @property
    def df(self):
        return jnp.exp(self.log_df)


class StudentT(Transformed):
    "Student T distribution with loc and scale."
    def __init__(self, df: Array, loc: Array, scale: Array):
        self.dim = df.shape[0]
        self.cond_dim = 0

        base_dist = StandardStudentT(df)
        bijection = Affine(loc, scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def scale(self):
        return self.bijection.scale

    @property
    def df(self):
        return self.base_dist.df

