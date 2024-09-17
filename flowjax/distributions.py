"""Distributions from flowjax.distributions."""

import inspect
from abc import abstractmethod
from collections.abc import Callable
from functools import wraps
from math import prod
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from equinox import AbstractVar
from jax import dtypes
from jax.nn import log_softmax, softplus
from jax.numpy import linalg
from jax.scipy import stats as jstats
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Shaped

from flowjax.bijections import (
    AbstractBijection,
    Affine,
    Chain,
    Exp,
    Scale,
    TriangularAffine,
)
from flowjax.utils import (
    _get_ufunc_signature,
    arraylike_to_array,
    inv_softplus,
    merge_cond_shapes,
)
from flowjax.wrappers import AbstractUnwrappable, Parameterize, unwrap


class AbstractDistribution(eqx.Module):
    """Abstract distribution class.

    Distributions are registered as JAX PyTrees (as they are equinox modules), and as
    such they are compatible with normal JAX operations.

    Concrete subclasses can be implemented as follows:

    - Inherit from :class:`AbstractDistribution`.
    - Define the abstract attributes ``shape`` and ``cond_shape``.
      ``cond_shape`` should be ``None`` for unconditional distributions.
    - Define the abstract method ``_sample`` which returns a single sample
      with shape ``dist.shape``, (given a single conditioning variable, if needed).
    - Define the abstract method ``_log_prob``, returning a scalar log probability
      of a single sample, (given a single conditioning variable, if needed).

    The abstract class then defines vectorized versions with shape checking for the
    public API. See the source code for :class:`StandardNormal` for a simple concrete
    example.

    Attributes:
        shape: Tuple denoting the shape of a single sample from the distribution.
        cond_shape: Tuple denoting the shape of an instance of the conditioning
            variable. This should be None for unconditional distributions.

    """

    shape: AbstractVar[tuple[int, ...]]
    cond_shape: AbstractVar[tuple[int, ...] | None]

    @abstractmethod
    def _log_prob(self, x: Array, condition: Array | None = None) -> Array:
        """Evaluate the log probability of point x.

        This method should be be valid for inputs with shapes matching
        ``distribution.shape`` and ``distribution.cond_shape`` for conditional
        distributions (i.e. it defines the method for unbatched inputs).
        """

    @abstractmethod
    def _sample(self, key: PRNGKeyArray, condition: Array | None = None) -> Array:
        """Sample a point from the distribution.

        This method should return a single sample with shape matching
        ``distribution.shape``.
        """

    def _sample_and_log_prob(self, key: PRNGKeyArray, condition: Array | None = None):
        """Sample a point from the distribution, and return its log probability."""
        x = self._sample(key, condition)
        return x, self._log_prob(x, condition)

    def log_prob(self, x: ArrayLike, condition: ArrayLike | None = None) -> Array:
        """Evaluate the log probability.

        Uses numpy-like broadcasting if additional leading dimensions are passed.

        Args:
            x: Points at which to evaluate density.
            condition: Conditioning variables. Defaults to None.

        Returns:
            Array: Jax array of log probabilities.
        """
        self = unwrap(self)
        x = arraylike_to_array(x, err_name="x", dtype=float)
        if self.cond_shape is not None:
            condition = arraylike_to_array(condition, err_name="condition", dtype=float)
        lps = self._vectorize(self._log_prob)(x, condition)
        return jnp.where(jnp.isnan(lps), -jnp.inf, lps)

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ) -> Array:
        """Sample from the distribution.

        For unconditional distributions, the output will be of shape
        ``sample_shape + dist.shape``. For conditional distributions, batch dimensions
        in the condition is supported, and the output will have shape
        ``sample_shape + condition_batch_shape + dist.shape``.

        Args:
            key: Jax random key.
            condition: Conditioning variables. Defaults to None.
            sample_shape: Sample shape. Defaults to ().
        """
        self = unwrap(self)
        if self.cond_shape is not None:
            condition = arraylike_to_array(condition, err_name="condition")
        keys = self._get_sample_keys(key, sample_shape, condition)
        return self._vectorize(self._sample)(keys, condition)

    def sample_and_log_prob(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        """Sample the distribution and return the samples with their log probabilities.

        For transformed distributions (especially flows), this will generally be more
        efficient than calling the methods seperately. Refer to the
        :py:meth:`~flowjax.distributions.AbstractDistribution.sample` documentation for
        more information.

        Args:
            key: Jax random key.
            condition: Conditioning variables. Defaults to None.
            sample_shape: Sample shape. Defaults to ().
        """
        self = unwrap(self)
        if self.cond_shape is not None:
            condition = arraylike_to_array(condition, err_name="condition")
        keys = self._get_sample_keys(key, sample_shape, condition)
        return self._vectorize(self._sample_and_log_prob)(keys, condition)

    @property
    def ndim(self) -> int:
        """Number of dimensions in the distribution (the length of the shape)."""
        return len(self.shape)

    @property
    def cond_ndim(self) -> None | int:
        """Number of dimensions of the conditioning variable (length of cond_shape)."""
        return None if self.cond_shape is None else len(self.cond_shape)

    def _vectorize(self, method: Callable) -> Callable:
        """Returns a vectorized version of the distribution method."""
        # Get shapes without broadcasting - note the (2, ) corresponds to key arrays.
        maybe_cond = [] if self.cond_shape is None else [self.cond_shape]
        in_shapes = {
            "_sample_and_log_prob": [()] + maybe_cond,
            "_sample": [()] + maybe_cond,
            "_log_prob": [self.shape] + maybe_cond,
        }
        out_shapes = {
            "_sample_and_log_prob": [self.shape, ()],
            "_sample": [self.shape],
            "_log_prob": [()],
        }
        in_shapes, out_shapes = in_shapes[method.__name__], out_shapes[method.__name__]

        def _check_shapes(method):
            # Wraps unvectorised method with shape checking
            @wraps(method)
            def _wrapper(*args, **kwargs):
                bound = inspect.signature(method).bind(*args, **kwargs)
                for in_shape, (name, arg) in zip(
                    in_shapes,
                    bound.arguments.items(),
                    strict=False,
                ):
                    if arg.shape != in_shape:
                        raise ValueError(
                            f"Expected trailing dimensions matching {in_shape} for "
                            f"{name}; got {arg.shape}.",
                        )
                return method(*args, **kwargs)

            return _wrapper

        signature = _get_ufunc_signature(in_shapes, out_shapes)
        ex = frozenset([1]) if self.cond_shape is None else frozenset()
        return jnp.vectorize(_check_shapes(method), signature=signature, excluded=ex)

    def _get_sample_keys(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...],
        condition,
    ):

        if not dtypes.issubdtype(key.dtype, dtypes.prng_key):
            raise TypeError("New-style typed JAX PRNG keys required.")

        if self.cond_ndim is not None:
            leading_cond_shape = condition.shape[: -self.cond_ndim or None]
        else:
            leading_cond_shape = ()
        key_shape = sample_shape + leading_cond_shape
        key_size = prod(key_shape)  # note: prod(()) == 1, so works for scalar smaples
        return jr.split(key, key_size).reshape(key_shape)


class AbstractTransformed(AbstractDistribution):
    """Abstract class respresenting transformed distributions.

    We take the forward bijection for use in sampling, and the inverse for use in
    density evaluation. See also :class:`Transformed`. Concete implementations should
    subclass :class:`AbstractTransformed`, and define the abstract attributes
    ``base_dist`` and ``bijection``. See the source code for :class:`Normal` as a
    simple example.

    .. warning::
            It is the users responsibility to ensure the bijection is valid across the
            entire support of the distribution. Failure to do so may result in
            non-finite values or incorrectly normalized densities.

    Attributes:
        base_dist: The base distribution.
        bijection: The transformation to apply.
    """

    base_dist: AbstractVar[AbstractDistribution]
    bijection: AbstractVar[AbstractBijection]

    def _log_prob(self, x, condition=None):
        z, log_abs_det = self.bijection.inverse_and_log_det(x, condition)
        p_z = self.base_dist._log_prob(z, condition)
        return p_z + log_abs_det

    def _sample(self, key, condition=None):
        base_sample = self.base_dist._sample(key, condition)
        return self.bijection.transform(base_sample, condition)

    def _sample_and_log_prob(
        self,
        key: PRNGKeyArray,
        condition: Array | None = None,
    ):  # TODO add overide decorator when python>=3.12 is common
        # We override to avoid computing the inverse transformation.
        base_sample, log_prob_base = self.base_dist._sample_and_log_prob(key, condition)
        sample, forward_log_dets = self.bijection.transform_and_log_det(
            base_sample,
            condition,
        )
        return sample, log_prob_base - forward_log_dets

    def __check_init__(self):  # TODO test errors and test conditional base distribution
        """Checks cond_shape is compatible in both bijection and distribution."""
        if (
            self.base_dist.cond_shape is not None
            and self.bijection.cond_shape is not None
            and self.base_dist.cond_shape != self.bijection.cond_shape
        ):
            raise ValueError(
                "The base distribution and bijection are both conditional "
                "but have mismatched cond_shape attributes. Base distribution has"
                f"{self.base_dist.cond_shape}, and the bijection has"
                f"{self.bijection.cond_shape}.",
            )

    def merge_transforms(self):
        """Unnests nested transformed distributions.

        Returns an equivilent distribution, but ravelling nested
        :class:`AbstractTransformed` distributions such that the returned distribution
        has a base distribution that is not an :class:`AbstractTransformed` instance.
        """
        if not isinstance(self.base_dist, AbstractTransformed):
            return self
        base_dist = self.base_dist
        bijections = [self.bijection]
        while isinstance(base_dist, AbstractTransformed):
            bijections.append(base_dist.bijection)
            base_dist = base_dist.base_dist
        bijection = Chain(list(reversed(bijections))).merge_chains()
        return Transformed(base_dist, bijection)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.base_dist.shape

    @property
    def cond_shape(self) -> tuple[int, ...] | None:
        return merge_cond_shapes((self.bijection.cond_shape, self.base_dist.cond_shape))


class Transformed(AbstractTransformed):
    """Form a distribution like object using a base distribution and a bijection.

    We take the forward bijection for use in sampling, and the inverse
    bijection for use in density evaluation.

    .. warning::
            It is the users responsibility to ensure the bijection is valid across the
            entire support of the distribution. Failure to do so may result in
            non-finite values or incorrectly normalized densities.

    Args:
        base_dist: Base distribution.
        bijection: Bijection to transform distribution.

    Example:
        .. doctest::

            >>> from flowjax.distributions import StandardNormal, Transformed
            >>> from flowjax.bijections import Affine
            >>> normal = StandardNormal()
            >>> bijection = Affine(1)
            >>> transformed = Transformed(normal, bijection)
    """

    base_dist: AbstractDistribution
    bijection: AbstractBijection


class AbstractLocScaleDistribution(AbstractTransformed):
    """Abstract distribution class for affine transformed distributions."""

    base_dist: AbstractVar[AbstractDistribution]
    bijection: AbstractVar[Affine]

    @property
    def loc(self):
        """Location of the distribution."""
        return self.bijection.loc

    @property
    def scale(self):
        """Scale of the distribution."""
        return unwrap(self.bijection.scale)


class StandardNormal(AbstractDistribution):
    """Standard normal distribution.

    Note unlike :class:`Normal`, this has no trainable parameters.

    Args:
        shape: The shape of the distribution. Defaults to ().
    """

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.norm.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.normal(key, self.shape)


class Normal(AbstractLocScaleDistribution):
    """An independent Normal distribution with mean and std for each dimension.

    ``loc`` and ``scale`` should broadcast to the desired shape of the distribution.

    Args:
        loc: Means. Defaults to 0. Defaults to 0.
        scale: Standard deviations. Defaults to 1.
    """

    base_dist: StandardNormal
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        self.base_dist = StandardNormal(
            jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc=loc, scale=scale)


class LogNormal(AbstractTransformed):
    """Log normal distribution.

    ``loc`` and ``scale`` here refers to the underlying normal distribution.

    Args:
        loc: Location paramter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: Normal
    bijection: Exp

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        self.base_dist = Normal(loc, scale)
        self.bijection = Exp(self.base_dist.shape)


class MultivariateNormal(AbstractTransformed):
    """Multivariate normal distribution.

    Internally this is parameterised using the Cholesky decomposition of the covariance
    matrix.

    Args:
        loc: The location/mean parameter vector. If this is scalar it is broadcast to
            the dimension implied by the covariance matrix.
        covariance: Covariance matrix.
    """

    base_dist: StandardNormal
    bijection: TriangularAffine

    def __init__(
        self,
        loc: Shaped[ArrayLike, "#dim"],
        covariance: Shaped[Array, "dim dim"],
    ):
        self.bijection = TriangularAffine(loc, linalg.cholesky(covariance))
        self.base_dist = StandardNormal(self.bijection.shape)

    @property
    def loc(self):
        """Location (mean) of the distribution."""
        return self.bijection.loc

    @property
    def covariance(self):
        """The covariance matrix."""
        cholesky = unwrap(self.bijection.triangular)
        return cholesky @ cholesky.T


class _StandardUniform(AbstractDistribution):
    r"""Standard Uniform distribution."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.uniform.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.uniform(key, shape=self.shape)


class Uniform(AbstractLocScaleDistribution):
    """Uniform distribution.

    ``minval`` and ``maxval`` should broadcast to the desired distribution shape.

    Args:
        minval: Minimum values.
        maxval: Maximum values.
    """

    base_dist: _StandardUniform
    bijection: Affine

    def __init__(self, minval: ArrayLike, maxval: ArrayLike):
        shape = jnp.broadcast_shapes(jnp.shape(minval), jnp.shape(maxval))
        minval, maxval = eqx.error_if(
            (minval, maxval), maxval <= minval, "minval must be less than the maxval."
        )
        self.base_dist = _StandardUniform(shape)
        self.bijection = Affine(loc=minval, scale=maxval - minval)

    @property
    def minval(self):
        """Minimum value of the uniform distribution."""
        return self.bijection.loc

    @property
    def maxval(self):
        """Maximum value of the uniform distribution."""
        return self.bijection.loc + unwrap(self.bijection.scale)


class _StandardGumbel(AbstractDistribution):
    """Standard gumbel distribution."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return -(x + jnp.exp(-x)).sum()

    def _sample(self, key, condition=None):
        return jr.gumbel(key, shape=self.shape)


class Gumbel(AbstractLocScaleDistribution):
    """Gumbel distribution.

    ``loc`` and ``scale`` should broadcast to the dimension of the distribution.

    Args:
        loc: Location paramter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: _StandardGumbel
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        self.base_dist = _StandardGumbel(
            jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc, scale)


class _StandardCauchy(AbstractDistribution):
    """Implements standard cauchy distribution (loc=0, scale=1)."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.cauchy.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.cauchy(key, shape=self.shape)


class Cauchy(AbstractLocScaleDistribution):
    """Cauchy distribution.

    ``loc`` and ``scale`` should broadcast to the dimension of the distribution.

    Args:
        loc: Location paramter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: _StandardCauchy
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        self.base_dist = _StandardCauchy(
            jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc, scale)


class _StandardStudentT(AbstractDistribution):
    """Implements student T distribution with specified degrees of freedom."""

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    df: Array | AbstractUnwrappable[Array]

    def __init__(self, df: ArrayLike):
        df = arraylike_to_array(df, dtype=float)
        df = eqx.error_if(df, df <= 0, "Degrees of freedom values must be positive.")
        self.shape = jnp.shape(df)
        self.df = Parameterize(softplus, inv_softplus(df))

    def _log_prob(self, x, condition=None):
        return jstats.t.logpdf(x, df=self.df).sum()

    def _sample(self, key, condition=None):
        return jr.t(key, df=self.df, shape=self.shape)


class StudentT(AbstractLocScaleDistribution):
    """Student T distribution.

    ``df``, ``loc`` and ``scale`` broadcast to the dimension of the distribution.

    Args:
        df: The degrees of freedom.
        loc: Location parameter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: _StandardStudentT
    bijection: Affine

    def __init__(self, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1):
        df, loc, scale = jnp.broadcast_arrays(df, loc, scale)
        self.base_dist = _StandardStudentT(df)
        self.bijection = Affine(loc, scale)

    @property
    def df(self):
        """The degrees of freedom of the distribution."""
        return unwrap(self.base_dist.df)


class _StandardLaplace(AbstractDistribution):
    """Implements standard laplace distribution (loc=0, scale=1)."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.laplace.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.laplace(key, shape=self.shape)


class Laplace(AbstractLocScaleDistribution):
    """Laplace distribution.

    ``loc`` and ``scale`` should broadcast to the dimension of the distribution.

    Args:
        loc: Location paramter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: _StandardLaplace
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        self.base_dist = _StandardLaplace(shape)
        self.bijection = Affine(loc, scale)


class _StandardExponential(AbstractDistribution):
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.expon.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.exponential(key, shape=self.shape)


class Exponential(AbstractTransformed):
    """Exponential distribution.

    Args:
        rate: The rate parameter (1 / scale).
    """

    base_dist: _StandardExponential
    bijection: Scale

    def __init__(self, rate: ArrayLike = 1):
        self.base_dist = _StandardExponential(jnp.shape(rate))
        self.bijection = Scale(1 / rate)

    @property
    def rate(self):
        return 1 / unwrap(self.bijection.scale)


class _StandardLogistic(AbstractDistribution):
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _sample(self, key, condition=None):
        return jr.logistic(key, self.shape)

    def _log_prob(self, x, condition=None):
        return jstats.logistic.logpdf(x).sum()


class Logistic(AbstractLocScaleDistribution):
    """Logistic distribution.

    ``loc`` and ``scale`` should broadcast to the shape of the distribution.

    Args:
        loc: Location parameter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: _StandardLogistic
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        self.base_dist = _StandardLogistic(
            shape=jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc=loc, scale=scale)


class VmapMixture(AbstractDistribution):
    """Create a mixture distribution.

    Given a distribution in which the arrays have a leading dimension with size matching
    the number of components, and a set of weights, create a mixture distribution.

    Example:
        .. doctest::

            >>> # Creating a 3 component, 2D gaussian mixture
            >>> from flowjax.distributions import Normal, VmapMixture
            >>> import equinox as eqx
            >>> import jax.numpy as jnp
            >>> normals = eqx.filter_vmap(Normal)(jnp.zeros((3, 2)))
            >>> mixture = VmapMixture(normals, weights=jnp.ones(3))
            >>> mixture.shape
            (2,)

    Args:
        dist: Distribution with a leading dimension in arrays with size equal to the
            number of mixture components. Often it is convenient to construct this with
            with a pattern like ``eqx.filter_vmap(MyDistribution)(my_params)``.
        weights: The positive, but possibly unnormalized component weights.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    log_normalized_weights: Array | AbstractUnwrappable[Array]
    dist: AbstractDistribution

    def __init__(
        self,
        dist: AbstractDistribution,
        weights: ArrayLike,
    ):
        weights = eqx.error_if(weights, weights <= 0, "Weights must be positive.")
        self.dist = dist
        self.log_normalized_weights = Parameterize(log_softmax, jnp.log(weights))
        self.shape = dist.shape
        self.cond_shape = dist.cond_shape

    def _log_prob(self, x, condition=None):
        log_probs = eqx.filter_vmap(lambda d: d._log_prob(x, condition))(self.dist)
        return logsumexp(log_probs + self.log_normalized_weights)

    def _sample(self, key, condition=None):
        key1, key2 = jr.split(key)
        component = jr.categorical(key1, self.log_normalized_weights)
        component_dist = tree_map(
            lambda leaf: leaf[component] if isinstance(leaf, Array) else leaf,
            tree=self.dist,
        )
        return component_dist._sample(key2, condition)
