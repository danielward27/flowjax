"""Distributions - including the base class Distribution, common distributions
and a Transformed distribution class.
"""
from abc import abstractmethod
from math import prod

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.experimental import checkify
from jax.scipy import stats as jstats
from jax.typing import ArrayLike

from flowjax.bijections import Affine, Bijection
from flowjax.utils import _get_ufunc_signature, arraylike_to_array, merge_cond_shapes


class Distribution(eqx.Module):
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

    Distributions are registered as jax PyTrees (as they are equinox modules), and as
    such they are compatible with normal jax operations.

    **Implementing a distribution**

        (1) Inherit from ``Distribution``.
        (2) Define attributes ``shape`` and ``cond_shape``. ``cond_shape`` should be
            ``None`` for unconditional distributions.
        (3) Define the ``_sample`` method, which samples a point with a shape of
            ``shape``, (given a conditioning variable with shape ``cond_shape`` for
            conditional distributions).
        (4) Define the ``_log_prob`` method, which evaluates the log probability,
            given an input of shape ``shape`` (and a conditioning variable with shape
            ``cond_shape`` for conditional distributions).

        The base class will handle defining more convenient log_prob and sample methods
        that support broadcasting and perform argument checks.

    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None

    @abstractmethod
    def _log_prob(self, x: Array, condition: Array | None = None) -> Array:
        """Evaluate the log probability of point x."""

    @abstractmethod
    def _sample(self, key: jr.KeyArray, condition: Array | None = None) -> Array:
        """Sample a point from the distribution."""

    def _sample_and_log_prob(
        self, key: jr.KeyArray, condition: Array | None = None
    ) -> tuple[Array, Array]:
        """Sample a point from the distribution, and return its log probability.
        Subclasses can reimplement this method in cases where more efficient methods
        exists (e.g. for :class:`Transformed` distributions).
        """
        x = self._sample(key, condition)
        log_prob = self._log_prob(x, condition)
        return x, log_prob

    def log_prob(self, x: ArrayLike, condition: ArrayLike | None = None) -> Array:
        """Evaluate the log probability. Uses numpy like broadcasting if additional
        leading dimensions are passed.

        Args:
            x (ArrayLike): Points at which to evaluate density.
            condition (ArrayLike | None): Conditioning variables. Defaults to None.

        Returns:
            Array: Jax array of log probabilities.
        """
        x = self._argcheck_and_cast_x(x)
        condition = self._argcheck_and_cast_condition(condition)
        if condition is None:
            sig = _get_ufunc_signature([self.shape], [()])
            exclude = frozenset([1])
        else:
            sig = _get_ufunc_signature([self.shape, self.cond_shape], [()])
            exclude = frozenset()

        lps = jnp.vectorize(self._log_prob, signature=sig, excluded=exclude)(
            x, condition
        )
        return jnp.where(jnp.isnan(lps), -jnp.inf, lps)  # type: ignore

    def sample(
        self,
        key: jr.KeyArray,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ) -> Array:
        """Sample from the distribution. For unconditional distributions, the output
        will be of shape ``sample_shape + dist.shape``. For conditional distributions,
        a batch dimension in the condition is supported, and the output shape will be
        ``sample_shape + condition_batch_shape + dist.shape``. See the example for more
        information.

        Args:
            key (jr.KeyArray): Jax random key.
            condition (ArrayLike | None): Conditioning variables. Defaults to None.
            sample_shape (tuple[int, ...]): Sample shape. Defaults to ().

        Example:
            The below example shows the behaviour of sampling, for an unconditional
            and a conditional distribution.

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
                cond_dist = CouplingFlow(
                    key, StandardNormal((2,)), cond_dim=3, transformer=Affine()
                    )

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


        """
        condition = self._argcheck_and_cast_condition(condition)
        excluded, signature = self._vectorize_sample_args()
        keys = self._get_sample_keys(key, sample_shape, condition)
        return jnp.vectorize(self._sample, excluded=excluded, signature=signature)(
            keys, condition
        )  # type: ignore

    def sample_and_log_prob(
        self,
        key: jr.KeyArray,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ):
        """Sample the distribution and return the samples and corresponding log
        probabilities. For transformed distributions (especially flows), this will
        generally be more efficient than calling the methods seperately. Refer to the
        :py:meth:`~flowjax.distributions.Distribution.sample` documentation for more
        information.

        Args:
            key (jr.KeyArray): Jax random key.
            condition (ArrayLike | None): Conditioning variables. Defaults to None.
            sample_shape (tuple[int, ...]): Sample shape. Defaults to ().
        """
        condition = self._argcheck_and_cast_condition(condition)

        excluded, signature = self._vectorize_sample_args(sample_and_log_prob=True)
        keys = self._get_sample_keys(key, sample_shape, condition)

        return jnp.vectorize(
            self._sample_and_log_prob, excluded=excluded, signature=signature
        )(keys, condition)

    def _vectorize_sample_args(self, sample_and_log_prob=False):
        """Get the excluded arguments and ufunc signature for sample or
        sample_and_log_prob"""
        out_shapes = [self.shape, ()] if sample_and_log_prob else [self.shape]
        if self.cond_shape is None:
            excluded = frozenset([1])
            in_shapes = [(2,)]
        else:
            excluded = frozenset()
            in_shapes = [(2,), self.cond_shape]
        signature = _get_ufunc_signature(in_shapes, out_shapes)
        return excluded, signature

    def _get_sample_keys(self, key, sample_shape, condition):
        """Splits a key into an arrray of keys with shape
        sample_shape + leading_cond_shape + (2,)."""
        if self.cond_shape is None:
            key_shape = sample_shape
        else:
            leading_cond_shape = (
                condition.shape[: -len(self.cond_shape)]
                if len(self.cond_shape) > 0
                else condition.shape
            )
            key_shape = sample_shape + leading_cond_shape

        key_size = max(1, prod(key_shape))  # Still need 1 key for scalar sample
        keys = jnp.reshape(jr.split(key, key_size), key_shape + (2,))  # type: ignore
        return keys

    def _argcheck_and_cast_x(self, x) -> Array:
        x = arraylike_to_array(x, err_name="x")
        x_trailing = x.shape[-self.ndim :] if self.ndim > 0 else ()
        if x_trailing != self.shape:
            raise ValueError(
                "Expected trailing dimensions in x to match the distribution shape "
                f"{self.shape}; got x shape {x.shape}."
            )
        return x

    def _argcheck_and_cast_condition(self, condition) -> Array | None:
        if self.cond_shape is None:
            if condition is not None:
                raise ValueError(
                    "Expected condition to be None for unconditional distribution; "
                    f"got {condition}."
                )
            return None
        condition = arraylike_to_array(condition, err_name="condition")
        condition_trailing = (
            condition.shape[-len(self.cond_shape) :] if self.cond_ndim > 0 else ()
        )
        if condition_trailing != self.cond_shape:
            raise ValueError(
                "Expected trailing dimensions in condition to match cond_shape "
                f"{self.cond_shape}, but got condition shape {condition.shape}."
            )
        return condition

    @property
    def ndim(self):
        """The number of dimensions in the distribution (the length of the shape)."""
        return len(self.shape)

    @property
    def cond_ndim(self):
        """The number of dimensions of the conditioning variable (length of
        cond_shape)."""
        if self.cond_shape is not None:
            return len(self.cond_shape)
        return None


class Transformed(Distribution):
    """Form a distribution like object using a base distribution and a
    bijection. We take the forward bijection for use in sampling, and the inverse
    bijection for use in density evaluation.
    """

    base_dist: Distribution
    bijection: Bijection
    cond_shape: tuple[int, ...] | None

    def __init__(
        self,
        base_dist: Distribution,
        bijection: Bijection,
    ):
        """
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
            It is the currently the users responsibility to ensure the bijection is
            valid across the entire support of the distribution. Failure to do so may
            lead to to unexpected results. In future versions explicit constraints may
            be introduced.
        """
        self.base_dist = base_dist
        self.bijection = bijection
        self.shape = self.base_dist.shape
        self.cond_shape = merge_cond_shapes(
            (self.bijection.cond_shape, self.base_dist.cond_shape)
        )

    def _log_prob(self, x, condition=None):
        z, log_abs_det = self.bijection.inverse_and_log_det(x, condition)
        p_z = self.base_dist._log_prob(z, condition)  # pylint: disable W0212
        return p_z + log_abs_det

    def _sample(self, key, condition=None):
        base_sample = self.base_dist._sample(key, condition)
        return self.bijection.transform(base_sample, condition)

    def _sample_and_log_prob(self, key: jr.KeyArray, condition=None):
        # We overwrite the naive implementation of calling both methods seperately to
        # avoid computing the inverse transformation.
        base_sample, log_prob_base = self.base_dist._sample_and_log_prob(key, condition)
        sample, forward_log_dets = self.bijection.transform_and_log_det(
            base_sample, condition
        )
        return sample, log_prob_base - forward_log_dets


class StandardNormal(Distribution):
    """Implements a standard normal distribution, condition is ignored. Unlike
    :class:`Normal`, this has no trainable parameters.
    """

    def __init__(self, shape: tuple[int, ...] = ()):
        """
        Args:
            shape (tuple[int, ...]): The shape of the normal distribution. Defaults to
                ().
        """
        self.shape = shape
        self.cond_shape = None

    def _log_prob(self, x, condition=None):
        return jstats.norm.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.normal(key, self.shape)


class Normal(Transformed):
    """Implements an independent Normal distribution with mean and std for
    each dimension. `loc` and `scale` should be broadcastable.
    """

    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """
        Args:
            loc (ArrayLike): Means. Defaults to 0.
            scale (ArrayLike): Standard deviations. Defaults to 1.
        """
        shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        base_dist = StandardNormal(shape)
        bijection = Affine(loc=loc, scale=scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        """Location of the distribution"""
        return self.bijection.loc

    @property
    def scale(self):
        """scale of the distribution"""
        return self.bijection.scale


class _StandardUniform(Distribution):
    r"""Implements a standard independent Uniform distribution, ie
    :math:`X \sim Uniform([0, 1]^d)`."""

    def __init__(self, shape: tuple[int, ...] = ()):
        self.shape = shape
        self.cond_shape = None

    def _log_prob(self, x, condition=None):
        return jstats.uniform.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.uniform(key, shape=self.shape)


class Uniform(Transformed):
    """Implements an independent uniform distribution between min and max for each
    dimension. `minval` and `maxval` should be broadcastable.
    """

    bijection: Affine

    def __init__(self, minval: ArrayLike, maxval: ArrayLike):
        """
        Args:
            minval (ArrayLike): Minimum values.
            maxval (ArrayLike): Maximum values.
        """
        minval, maxval = arraylike_to_array(minval), arraylike_to_array(maxval)
        shape = jnp.broadcast_shapes(minval.shape, maxval.shape)
        checkify.check(
            jnp.all(maxval >= minval),
            "Minimums must be less than the maximums.",
        )

        base_dist = _StandardUniform(shape)
        bijection = Affine(loc=minval, scale=maxval - minval)
        super().__init__(base_dist, bijection)

    @property
    def minval(self):
        """Minimum value of the uniform distribution."""
        return self.bijection.loc

    @property
    def maxval(self):
        """Maximum value of the uniform distribution."""
        return self.bijection.loc + self.bijection.scale


class _StandardGumbel(Distribution):
    """Standard gumbel distribution (https://en.wikipedia.org/wiki/Gumbel_distribution)."""

    def __init__(self, shape: tuple[int, ...] = ()):
        self.shape = shape
        self.cond_shape = None

    def _log_prob(self, x, condition=None):
        return -(x + jnp.exp(-x)).sum()

    def _sample(self, key, condition=None):
        return jr.gumbel(key, shape=self.shape)


class Gumbel(Transformed):
    """Gumbel distribution (https://en.wikipedia.org/wiki/Gumbel_distribution)"""

    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """
        `loc` and `scale` should broadcast to the dimension of the distribution.

        Args:
            loc (ArrayLike): Location paramter.
            scale (ArrayLike): Scale parameter. Defaults to 1.0.
        """
        shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        base_dist = _StandardGumbel(shape)
        bijection = Affine(loc, scale)

        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        """Location of the distribution"""
        return self.bijection.loc

    @property
    def scale(self):
        """Scale of the distribution."""
        return self.bijection.scale


class _StandardCauchy(Distribution):
    """
    Implements standard cauchy distribution (loc=0, scale=1)
    Ref: https://en.wikipedia.org/wiki/Cauchy_distribution
    """

    def __init__(self, shape: tuple[int, ...] = ()):
        self.shape = shape
        self.cond_shape = None

    def _log_prob(self, x, condition=None):
        return jstats.cauchy.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.cauchy(key, shape=self.shape)


class Cauchy(Transformed):
    """Cauchy distribution (https://en.wikipedia.org/wiki/Cauchy_distribution)."""

    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """
        `loc` and `scale` should broadcast to the dimension of the distribution.

        Args:
            loc (ArrayLike): Location paramter.
            scale (ArrayLike): Scale parameter. Defaults to 1.0.
        """
        shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        base_dist = _StandardCauchy(shape)
        bijection = Affine(loc, scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        """Location of the distribution"""
        return self.bijection.loc

    @property
    def scale(self):
        """scale of the distribution"""
        return self.bijection.scale


class _StandardStudentT(Distribution):
    """Implements student T distribution with specified degrees of freedom."""

    log_df: Array

    def __init__(self, df: ArrayLike):
        self.shape = jnp.shape(df)
        self.cond_shape = None
        self.log_df = jnp.log(df)

    def _log_prob(self, x, condition=None):
        return jstats.t.logpdf(x, df=self.df).sum()

    def _sample(self, key, condition=None):
        return jr.t(key, df=self.df, shape=self.shape)

    @property
    def df(self):
        """The degrees of freedom of the distibution."""
        return jnp.exp(self.log_df)


class StudentT(Transformed):
    """Student T distribution (https://en.wikipedia.org/wiki/Student%27s_t-distribution)."""

    bijection: Affine
    base_dist: _StandardStudentT

    def __init__(self, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """
        `df`, `loc` and `scale` broadcast to the dimension of the distribution.

        Args:
            df (ArrayLike): The degrees of freedom.
            loc (ArrayLike): Location parameter. Defaults to 0.0.
            scale (ArrayLike): Scale parameter. Defaults to 1.0.
        """
        df, loc, scale = jnp.broadcast_arrays(df, loc, scale)
        base_dist = _StandardStudentT(df)
        bijection = Affine(loc, scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        """Location of the distribution"""
        return self.bijection.loc

    @property
    def scale(self):
        """scale of the distribution"""
        return self.bijection.scale

    @property
    def df(self):
        """The degrees of freedom of the distribution."""
        return self.base_dist.df
