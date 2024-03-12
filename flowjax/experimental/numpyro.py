"""Utilities for interfacing with numpyro.

Note these utilities require `numpyro <https://github.com/pyro-ppl/numpyro>`_ to be
installed.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
from jax import Array
from jax.typing import ArrayLike

from flowjax import wrappers
from flowjax.bijections import AbstractBijection
from flowjax.distributions import AbstractDistribution, AbstractTransformed
from flowjax.utils import _VectorizedBijection, arraylike_to_array

try:
    import numpyro
except ImportError as e:
    e.add_note(
        "Note, in order to interface with numpyro, it must be installed. Please see "
        "https://num.pyro.ai/en/latest/getting_started.html#installation",
    )
    raise

from numpyro.distributions.constraints import (
    _IndependentConstraint,
    _Real,
)
from numpyro.distributions.transforms import IndependentTransform, biject_to

PyTree = Any


class _RealNdim(_IndependentConstraint):
    def __init__(self, event_dim: int):
        super().__init__(_Real(), event_dim)


@biject_to.register(_RealNdim)
def _biject_to_independent(constraint):
    return IndependentTransform(
        biject_to(constraint.base_constraint),
        constraint.reinterpreted_batch_ndims,
    )


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
):
    """Register numpyro params for an arbitrary pytree.

    This partitions the parameters and static components, registers the parameters using
    numpyro.param, then recombines them. This should be called from within an inference
    context to have an effect, e.g. within a numpyro model or guide function.

    Args:
        name: Name for the parameter set.
        model: The pytree (e.g. an equinox module, flowjax distribution/bijection).
    """
    params, static = eqx.partition(
        model,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, wrappers.NonTrainable),
    )
    if callable(params):
        # Wrap to avoid special handling of callables by numpyro. Numpyro expects a
        # callable to be used for lazy initialization, whereas in our case it is likely
        # a callable module we wish to train.
        params = numpyro.param(name, lambda _: params)
    else:
        params = numpyro.param(name, params)
    return wrappers.unwrap(eqx.combine(params, static))


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
        return _transformed_to_numpyro(dist, condition)
    return _DistributionToNumpyro(dist, condition)


class _DistributionToNumpyro(numpyro.distributions.Distribution):
    """Convert a AbstractDistribution to a numpyro distribution.

    Note that for transformed distributions, ``_transformed_to_numpyro`` should be used
    instead.
    """

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
        self.support = _RealNdim(dist.ndim)
        batch_shape = _get_batch_shape(condition, dist.cond_shape)
        super().__init__(batch_shape, dist.shape)

    @property
    def condition(self):
        return jax.lax.stop_gradient(self._condition)

    def sample(self, key, sample_shape=()):
        return self.dist.sample(key, sample_shape, self.condition)

    def log_prob(self, value):
        return self.dist.log_prob(value, self.condition)


def _transformed_to_numpyro(dist, condition=None):
    dist = dist.merge_transforms()  # Ensure base dist not transformed

    if condition is not None:
        condition = arraylike_to_array(condition, "condition")

    if dist.base_dist.cond_shape is not None:
        base_dist = _DistributionToNumpyro(dist.base_dist, condition)
    else:
        # add batch dimension to the base dist if condition is batched
        batch_shape = _get_batch_shape(condition, dist.cond_shape)
        base_dist = _DistributionToNumpyro(dist.base_dist).expand(batch_shape)

    transform = _BijectionToNumpyro(dist.bijection, condition)
    return numpyro.distributions.TransformedDistribution(base_dist, transform)


def _get_batch_shape(condition, cond_shape):
    if condition is not None:
        if len(cond_shape) > 0:
            return condition.shape[: -len(cond_shape)]
        return condition.shape
    return ()


class _BijectionToNumpyro(numpyro.distributions.transforms.Transform):
    """Wrap a numpyro AbstractBijection to a numpyro transform."""

    def __init__(
        self,
        bijection: AbstractBijection,
        condition: Array = None,
        domain=None,
        codomain=None,
    ):
        self.bijection = bijection
        self._condition = condition

        if domain is None:
            domain = _RealNdim(len(bijection.shape))
        if codomain is None:
            codomain = _RealNdim(len(bijection.shape))
        self.domain = domain
        self.codomain = codomain
        self._argcheck_domains()

    def __call__(self, x):
        return self.vbijection.transform(x, self.condition)

    def _inverse(self, y):
        return self.vbijection.inverse(y, self.condition)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        if intermediates is not None:
            return intermediates  # Logdet calculated with forward transformation
        return self.vbijection.transform_and_log_det(x, self.condition)[1]

    def call_with_intermediates(self, x):
        return self.vbijection.transform_and_log_det(x, self.condition)

    @property
    def condition(self):
        return jax.lax.stop_gradient(self._condition)

    @property
    def vbijection(self):
        return _VectorizedBijection(self.bijection)

    def tree_flatten(self):
        raise NotImplementedError()

    def _argcheck_domains(self):
        for k, v in {"domain": self.domain, "codomain": self.codomain}.items():
            if v is not None and v.event_dim != len(self.bijection.shape):
                raise ValueError(
                    f"{k}.event_dim {v.event_dim} did not match the length of the "
                    f"bijection shape {len(self.bijection.shape)}.",
                )
