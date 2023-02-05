from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp

import jax.random as jr
from jax.scipy import stats as jstats

from flowjax.bijections import Affine, Bijection
from flowjax.utils import Array, _get_ufunc_signature
from math import prod
from flowjax.utils import merge_shapes

from jax.experimental import checkify

class Distribution(eqx.Module, ABC):
    """Distribution base class. Distributions all have an attribute ``shape``,
    denoting the shape of a single sample from the distribution. This corresponds to the 
    ``batch_shape + event_shape`` in torch/numpyro distributions. Similarly, the
    ``cond_shape`` attribute denotes the shape of the conditioning variable.
    This attribute is None for unconditional distributions. For example

    .. doctest::

        >>> import jax.numpy as jnp
        >>> from flowjax.distributions import Normal
        >>> dist = Normal(jnp.zeros(2))
        >>> dist.shape
        (2,)
        >>> dist.cond_shape is None
        True

    Distributions are registered as jax PyTrees (as they are equinox modules), and as such
    they are compatible with normal jax operations.

    Implementing a distribution

        (1) Inherit from ``Distribution``.
        (2) Define attributes ``shape`` and ``cond_shape`` (cond shape should be None for unconditional distributions).
        (3) Define the ``_sample`` method, which samples a point with a shape of ``shape``, (given a conditioning variable with shape ``cond_shape`` for conditional distributions).
        (4) Define the ``_log_prob`` method, which evaluates the log probability, given an input of shape ``shape`` (and a conditioning variable with shape ``cond_shape`` for conditional distributions).

        The base class will handle defining more convenient log_prob and sample methods that support broadcasting and perform argument checks.

    """

    shape: Tuple[int]
    cond_shape: Union[None, Tuple[int]]

    @abstractmethod
    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        "Evaluate the log probability of point x."
        pass

    @abstractmethod
    def _sample(self, key: jr.KeyArray, condition: Optional[Array] = None):
        "Sample a point from the distribution."
        pass

    def _sample_and_log_prob(
        self,
        key: jr.PRNGKey,
        condition: Optional[Array] = None
        ):
        """
        Sample a point from the distribution, and return its log probability.
        Subclasses can reimplement this method in cases where more efficient methods exists (e.g. see Transformed).
        """
        x = self._sample(key, condition)
        log_prob = self._log_prob(x, condition)
        return x, log_prob

    def log_prob(self, x: Array, condition: Optional[Array] = None):
        """Evaluate the log probability. Uses numpy like broadcasting if additional
        leading dimensions are passed.

        Args:
            x (Array): Points at which to evaluate density.
            condition (Optional[Array], optional): Conditioning variables. Defaults to None.

        Returns:
            Array: Jax array of log probabilities.
        """
        self._argcheck(x, condition)
        if condition is not None:
            sig = _get_ufunc_signature([self.shape, self.cond_shape], [()])
            exclude = {}
        else:
            sig = _get_ufunc_signature([self.shape], [()])
            exclude = {1}
        lps = jnp.vectorize(self._log_prob, signature=sig, excluded=exclude)(
            x, condition
        )
        return jnp.where(jnp.isnan(lps), -jnp.inf, lps)

    def sample(
        self,
        key: jr.PRNGKey,
        sample_shape: Tuple[int] = (),
        condition: Optional[Array] = None,
    ):
        """Sample from the distribution. For unconditional distributions, the output will
        be of shape ``sample_shape + dist.shape``.
        
        Example:

            .. testsetup::

                from flowjax.distributions import StandardNormal
                import jax.random as jr
                import jax.numpy as jnp
                from flowjax.flows import CouplingFlow
                from flowjax.bijections import Affine
                # For a unconditional distribution:
                key = jr.PRNGKey(0)
                dist = StandardNormal((2,))
                # For a conditional distribution
                cond_dist = CouplingFlow(key, StandardNormal((2,)), cond_dim=3, transformer=Affine())

            For an unconditional distribution:

            .. doctest::

                >>> dist.shape
                (2,)
                >>> samples = dist.sample(key, (10, ))
                >>> samples.shape
                (10, 2)

            For a conditional distribution:

            .. doctest::

                >>> cond_dist.shape
                (2,)
                >>> cond_dist.cond_shape
                (3,)
                >>> # Sample 10 times for a particular condition
                >>> samples = cond_dist.sample(key, (10,), condition=jnp.ones(3))
                >>> samples.shape
                (10, 2)
                >>> # Sampling, batching over a condition
                >>> samples = cond_dist.sample(key, condition=jnp.ones((5, 3)))
                >>> samples.shape
                (5, 2)
                >>> # Sample 10 times for each of 5 conditioning variables
                >>> samples = cond_dist.sample(key, (10,), condition=jnp.ones((5, 3)))
                >>> samples.shape
                (10, 5, 2)

        Args:
            key (jr.PRNGKey): Jax random key.
            condition (Optional[Array], optional): Conditioning variables. Defaults to None.
            sample_shape (Tuple[int], optional): Sample shape. Defaults to ().

        """
        self._argcheck(condition=condition)

        if condition is None:
            key_shape = sample_shape
            excluded = {1}
            sig = _get_ufunc_signature([(2,)], [self.shape])
        else:
            leading_cond_shape = condition.shape[: -len(self.cond_shape)] if self.cond_ndim > 0 else condition.shape
            key_shape = sample_shape + leading_cond_shape
            excluded = {}
            sig = _get_ufunc_signature([(2,), self.cond_shape], [self.shape])

        key_size = max(1, prod(key_shape))  # Still need 1 key for scalar input
        keys = jnp.reshape(jr.split(key, key_size), key_shape + (2,))

        return jnp.vectorize(self._sample, excluded=excluded, signature=sig)(
            keys, condition
        )


    def sample_and_log_prob(
        self,
        key: jr.PRNGKey,
        sample_shape: Tuple[int] = (),
        condition: Optional[Array] = None,
        ):
        """Sample the distribution and return the samples and corresponding log probabilities.
        For transformed distributions (especially flows), this will generally be more efficient
        than calling the methods seperately.
        
        Refer to the :py:meth:`~flowjax.distributions.Distribution.sample` and
        Refer to the :py:meth:`~flowjax.distributions.Distribution.log_prob` documentation
        for more information.

        Args:
            key (jr.PRNGKey): Jax random key.
            condition (Optional[Array], optional): Conditioning variables. Defaults to None.
            sample_shape (Tuple[int], optional): Sample shape. Defaults to ().
        """
        self._argcheck(condition=condition)

        if condition is None:
            key_shape = sample_shape
            excluded = {1}
            sig = _get_ufunc_signature([(2,)], [self.shape, ()])
        else:
            leading_cond_shape = condition.shape[: -len(self.cond_shape)] if self.cond_ndim > 0 else condition.shape
            key_shape = sample_shape + leading_cond_shape
            excluded = {}
            sig = _get_ufunc_signature([(2,), self.cond_shape], [self.shape, ()])

        key_size = max(1, prod(key_shape))  # Still need 1 key for scalar input
        keys = jnp.reshape(jr.split(key, key_size), key_shape + (2,))

        return jnp.vectorize(self._sample_and_log_prob, excluded=excluded, signature=sig)(
            keys, condition
        )

    def _argcheck(self, x=None, condition=None):
        # jnp.vectorize would catch ndim mismatches, but it doesn't check axis lengths.
        if x is not None:
            x_trailing = x.shape[-self.ndim :] if self.ndim > 0 else ()
            if x_trailing != self.shape:
                raise ValueError(
                    f"Expected trailing dimensions in input x to match the distribution shape, but got "
                    f"x shape {x.shape}, and distribution shape {self.shape}."
                )

        if condition is None and self.cond_shape is not None:
            raise ValueError(
                f"Conditioning variable was not provided. "
                f"Expected conditioning variable with trailing shape {self.shape}."
            )

        if condition is not None:
            if self.cond_shape is None:
                raise ValueError("condition should not be provided for unconditional distribution.")
            else:
                condition_trailing = (
                    condition.shape[-self.cond_ndim :] if self.cond_ndim > 0 else ()
                )
                if condition_trailing != self.cond_shape:
                    raise ValueError(
                        f"Expected trailing dimensions in the condition to match distribution.cond_shape, but got "
                        f"condition shape {condition.shape}, and distribution.cond_shape {self.cond_shape}."
                    )

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def cond_ndim(self):
        return len(self.cond_shape)


class Transformed(Distribution):
    base_dist: Distribution
    bijection: Bijection
    cond_shape: Union[None, Tuple[int]]

    def __init__(
        self,
        base_dist: Distribution,
        bijection: Bijection,
    ):
        """
        Form a distribution like object using a base distribution and a
        bijection. We take the forward bijection for use in sampling, and the inverse
        bijection for use in density evaluation.

        Args:
            base_dist (Distribution): Base distribution.
            bijection (Bijection): Bijection to transform distribution.

        Example:

        .. doctest::

            >>> from flowjax.distributions import StandardNormal, Transformed
            >>> from flowjax.bijections import Affine
            >>> normal = StandardNormal()
            >>> bijection = Affine(1)
            >>> transformed = Transformed(normal, bijection)

        
        .. warning::
            It is the currently the users responsibility to ensure the bijection is valid
            across the entire support of the distribution. Failure to do so may lead to
            to unexpected results. In future versions explicit constraints may be introduced.
        """
        self.base_dist = base_dist
        self.bijection = bijection
        self.shape = self.base_dist.shape
        self.cond_shape = merge_shapes(
            (self.bijection.cond_shape, self.base_dist.cond_shape)
        )

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        z, log_abs_det = self.bijection.inverse_and_log_abs_det_jacobian(x, condition)
        p_z = self.base_dist._log_prob(z, condition)
        return p_z + log_abs_det

    def _sample(self, key: jr.KeyArray, condition: Optional[Array] = None):
        z = self.base_dist._sample(key, condition)
        x = self.bijection.transform(z, condition)
        return x

    def _sample_and_log_prob(self, key: jr.PRNGKey, condition: Optional[Array] = None):
        # We overwrite the naive implementation of calling both methods seperately to
        # avoid computing the inverse transformation.
        x, log_prob_base = self.base_dist._sample_and_log_prob(key, condition)
        y, forward_log_dets = self.bijection.transform_and_log_abs_det_jacobian(x, condition)
        return y, log_prob_base - forward_log_dets


class StandardNormal(Distribution):
    def __init__(self, shape: Tuple[int] = ()):
        """
        Implements a standard normal distribution, condition is ignored.

        Args:
            shape (Tuple[int]): The shape of the normal distribution. Defaults to ().
        """
        self.shape = shape
        self.cond_shape = None

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.norm.logpdf(x).sum()

    def _sample(self, key: jr.KeyArray, condition: Optional[Array] = None):
        return jr.normal(key, self.shape)


class Normal(Transformed):
    """
    Implements an independent Normal distribution with mean and std for
    each dimension. `loc` and `scale` should be broadcastable.
    """

    def __init__(self, loc: Array=0, scale: Array=1):
        """
        Args:
            loc (Array): Means. Defaults to 0.
            scale (Array): Standard deviations. Defaults to 1.
        """
        self.shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        self.cond_shape = None
        base_dist = StandardNormal(self.shape)
        bijection = Affine(loc=loc, scale=scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def scale(self):
        return self.bijection.scale


class _StandardUniform(Distribution):
    """
    Implements a standard independent Uniform distribution, ie X ~ Uniform([0, 1]^dim).
    """

    def __init__(self, shape: Tuple[int] = ()):
        self.shape = shape
        self.cond_shape = None

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.uniform.logpdf(x).sum()

    def _sample(self, key: jr.KeyArray, condition: Optional[Array] = None):
        return jr.uniform(key, shape=self.shape)




class Uniform(Transformed):
    """
    Implements an independent uniform distribution
    between min and max for each dimension. `minval` and `maxval` should be broadcastable.
    """

    def __init__(self, minval: Array, maxval: Array):
        """
        Args:
            minval (Array): Minimum values.
            maxval (Array): Maximum values.
        """
        self.shape = jnp.broadcast_shapes(jnp.shape(minval), jnp.shape(maxval))
        self.cond_shape = None

        checkify.check(
            jnp.all(maxval >= minval), "Minimums must be less than the maximums."
        )

        base_dist = _StandardUniform(self.shape)
        bijection = Affine(loc=minval, scale=maxval - minval)
        super().__init__(base_dist, bijection)

    @property
    def minval(self):
        return self.bijection.loc

    @property
    def maxval(self):
        return self.bijection.loc + self.bijection.scale


class _StandardGumbel(Distribution):
    """Standard gumbel distribution (https://en.wikipedia.org/wiki/Gumbel_distribution)."""

    def __init__(self, shape: Tuple[int] = ()):
        self.shape = shape
        self.cond_shape = None

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return -(x + jnp.exp(-x)).sum()

    def _sample(self, key: jr.KeyArray, condition: Optional[Array] = None):
        return jr.gumbel(key, shape=self.shape)


class Gumbel(Transformed):
    """Gumbel distribution (https://en.wikipedia.org/wiki/Gumbel_distribution)"""

    def __init__(self, loc: Array=0, scale: Array=1):
        """
        `loc` and `scale` should broadcast to the dimension of the distribution.

        Args:
            loc (Array): Location paramter.
            scale (Array, optional): Scale parameter. Defaults to 1.0.
        """
        self.shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        self.cond_shape = None
        base_dist = _StandardGumbel(self.shape)
        bijection = Affine(loc, scale)

        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def scale(self):
        return self.bijection.scale


class _StandardCauchy(Distribution):
    """
    Implements standard cauchy distribution (loc=0, scale=1)
    Ref: https://en.wikipedia.org/wiki/Cauchy_distribution
    """

    def __init__(self, shape: Tuple[int] = ()):
        self.shape = shape
        self.cond_shape = None

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.cauchy.logpdf(x).sum()

    def _sample(self, key: jr.KeyArray, condition: Optional[Array] = None):
        return jr.cauchy(key, shape=self.shape)


class Cauchy(Transformed):
    """
    Cauchy distribution (https://en.wikipedia.org/wiki/Cauchy_distribution).
    """

    def __init__(self, loc: Array=0, scale: Array=1):
        """
        `loc` and `scale` should broadcast to the dimension of the distribution.

        Args:
            loc (Array): Location paramter.
            scale (Array, optional): Scale parameter. Defaults to 1.0.
        """
        self.shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        self.cond_shape = None
        base_dist = _StandardCauchy(self.shape)
        bijection = Affine(loc, scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def scale(self):
        return self.bijection.scale


class _StandardStudentT(Distribution):
    """
    Implements student T distribution with specified degrees of freedom.
    """

    log_df: Array

    def __init__(self, df: Array):
        self.shape = df.shape
        self.cond_shape = None
        self.log_df = jnp.log(df)

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.t.logpdf(x, df=self.df).sum()

    def _sample(self, key: jr.KeyArray, condition: Optional[Array] = None):
        return jr.t(key, df=self.df, shape=self.shape)

    @property
    def df(self):
        return jnp.exp(self.log_df)


class StudentT(Transformed):
    """Student T distribution (https://en.wikipedia.org/wiki/Student%27s_t-distribution)."""

    def __init__(self, df: Array, loc: Array=0, scale: Array=1):
        """
        `df`, `loc` and `scale` broadcast to the dimension of the distribution.

        Args:
            df (Array): The degrees of freedom.
            loc (Array): Location parameter. Defaults to 0.0.
            scale (Array, optional): Scale parameter. Defaults to 1.0.
        """
        df, loc, scale = jnp.broadcast_arrays(df, loc, scale)
        self.shape = df.shape
        self.cond_shape = None
        base_dist = _StandardStudentT(df)
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
