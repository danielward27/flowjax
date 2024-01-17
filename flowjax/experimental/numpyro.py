"""Utilities for interfacing with numpyro.

Note these utilities require `numpyro <https://github.com/pyro-ppl/numpyro>`_ to be
installed.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from flowjax.utils import arraylike_to_array

try:
    import numpyro
except ImportError as e:
    e.add_note(
        "Note, in order to interface with numpyro, it must be installed. Please see "
        "https://num.pyro.ai/en/latest/getting_started.html#installation",
    )
    raise

from numpyro.distributions import constraints

from flowjax.bijections import AbstractBijection
from flowjax.distributions import AbstractDistribution, AbstractTransformed
from flowjax.utils import _get_ufunc_signature

PyTree = Any


def sample(name: str, fn: Any, *args, condition=None, **kwargs):
    """Numpyro sample wrapper that wraps flowjax distributions.

    Args:
        name: Name of the sample site.
        fn: A flowjax distribution, numpyro distribution or a stochastic function that
            returns a sample.
        condition: Conditioning variable if fn is a conditional flowjax distribution.
            Defaults to None.
        *args: Passed to numpyro sample.
        **kwargs: Passed to numpyro sample.
    """
    if isinstance(fn, AbstractDistribution):
        fn = distribution_to_numpyro(fn, condition)

    return numpyro.sample(name, fn, *args, **kwargs)


def register_params(
    name: str,
    model: PyTree,
    filter_spec: Callable | PyTree = eqx.is_inexact_array,
):
    """Register numpyro params for an arbitrary pytree.

    This partitions the parameters and static components, registers the parameters using
    numpyro.param, then recombines them. This should be called from within an inference
    context to have an effect, e.g. within a numpyro model or guide function.

    Args:
        name: Name for the parameter set.
        model: The pytree (e.g. an equinox module, flowjax distribution/bijection).
        filter_spec: Equinox `filter_spec` for specifying trainable parameters. Either a
            callable `leaf -> bool`, or a PyTree with prefix structure matching `dist`
            with True/False values. Defaults to `eqx.is_inexact_array`.

    """
    params, static = eqx.partition(model, filter_spec)
    if callable(params):
        # Wrap to avoid special handling of callables by numpyro. Numpyro expects a
        # callable to be used for lazy initialization, whereas in our case it is likely
        # a callable module we wish to train.
        params = numpyro.param(name, lambda _: params)
    else:
        params = numpyro.param(name, params)
    return eqx.combine(params, static)


def distribution_to_numpyro(
    dist: AbstractDistribution,
    condition: ArrayLike | None = None,
):
    """Convert a flowjax distribution to a numpyro distribution.

    Args:
        dist (AbstractDistribution): Flowjax distribution
        condition: condition: Conditioning variables. Any leading batch dimensions will
            be converted to batch dimensions in the numpyro distribution. Defaults to
            None.
    """
    if isinstance(dist, AbstractTransformed):
        return _TransformedToNumpyro(dist, condition)
    return _DistributionToNumpyro(dist, condition)


class _DistributionToNumpyro(numpyro.distributions.Distribution):
    dist: AbstractDistribution
    _condition: Array

    def __init__(
        self,
        dist: AbstractDistribution,
        condition: ArrayLike | None = None,
    ):
        self.dist = dist

        if condition is not None:
            condition = arraylike_to_array(condition, "condition")
        self._condition = condition
        self.support = constraints.real
        batch_shape = _get_batch_shape(condition, dist.cond_shape)
        super().__init__(batch_shape=batch_shape, event_shape=dist.shape)

    @property
    def condition(self):
        return jax.lax.stop_gradient(self._condition)

    def sample(self, key, sample_shape=...):
        return self.dist.sample(key, sample_shape, self.condition)

    def log_prob(self, value):
        return self.dist.log_prob(value, self.condition)


class _TransformedToNumpyro(numpyro.distributions.Distribution):
    def __init__(
        self,
        dist: AbstractTransformed,
        condition: ArrayLike | None = None,
    ):
        self.dist = dist.merge_transforms()  # Ensure base distribution not transformed
        if condition is not None:
            condition = arraylike_to_array(condition, "condition")
        self._condition = condition
        self.support = constraints.real
        batch_shape = _get_batch_shape(condition, dist.cond_shape)
        super().__init__(batch_shape=batch_shape, event_shape=dist.shape)

    @property
    def condition(self):
        return jax.lax.stop_gradient(self._condition)

    def sample(self, key, sample_shape=...):
        return self.dist.sample(key, sample_shape, self.condition)

    def sample_with_intermediates(self, key, sample_shape=...):
        # Sample the distribution returning the base distribution sample.
        z = self.dist.base_dist.sample(key, sample_shape, self._base_condition)
        x, log_det = _VectorizedBijection(self.dist.bijection).transform_and_log_det(
            z, self.condition
        )
        return x, [z, log_det]

    def log_prob(self, value, intermediates=None):
        if intermediates is None:
            return self.dist.log_prob(value, self.condition)

        z, log_det = intermediates
        return self.dist.base_dist.log_prob(z, self._base_condition) - log_det

    @property
    def _base_condition(self):
        return self.condition if self.dist.base_dist.cond_shape else None


class _VectorizedBijection:
    """Wrap a flowjax bijection to support vectorization.

    Args:
        bijection: flowjax bijection to be wrapped.
    """

    def __init__(self, bijection: AbstractBijection):
        self.bijection = bijection
        self.shape = self.bijection.shape
        self.cond_shape = self.bijection.cond_shape

    def transform(self, x, condition=None):
        transform = self.vectorize(self.bijection.transform)
        return transform(x, condition)

    def inverse(self, y, condition=None):
        inverse = self.vectorize(self.bijection.inverse)
        return inverse(y, condition)

    def transform_and_log_det(self, x, condition=None):
        transform_and_log_det = self.vectorize(
            self.bijection.transform_and_log_det,
            log_det=True,
        )
        return transform_and_log_det(x, condition)

    def vectorize(self, func, *, log_det=False):
        in_shapes, out_shapes = [self.bijection.shape], [self.bijection.shape]
        if log_det:
            out_shapes.append(())
        if self.bijection.cond_shape is not None:
            in_shapes.append(self.bijection.cond_shape)
            exclude = frozenset()
        else:
            exclude = frozenset([1])
        sig = _get_ufunc_signature(in_shapes, out_shapes)
        return jnp.vectorize(func, signature=sig, excluded=exclude)


def _get_batch_shape(condition, cond_shape):
    if condition is not None:
        if len(cond_shape) > 0:
            return condition.shape[: -len(cond_shape)]
        return condition.shape
    return ()
