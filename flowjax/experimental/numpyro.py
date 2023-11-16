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
from flowjax.distributions import AbstractTransformed
from flowjax.utils import _get_ufunc_signature

PyTree = Any
# TODO list:
#    - How to add support of batch dimensions.
#    - Do I need to support non-transformed distributions?
#    - Allow control of supports and constraints - will applications of transformations
#           to apply constraints lead to problems with reparameterisation?


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


class TransformedToNumpyro(numpyro.distributions.Distribution):
    """Convert a flowjax transformed distribution to a numpyro distribution.

    We assume the support of the distribution is unbounded.

    Args:
        dist: The flowjax distribution.
        condition: Conditioning variables. Any leading batch dimensions will be
            converted to batch dimensions in the numpyro distribution. Defaults to None.
    """

    def __init__(
        self,
        dist: AbstractTransformed,
        condition: ArrayLike | None = None,
    ):
        if condition is not None:
            condition = arraylike_to_array(condition, "condition")
            batch_shape = (
                condition.shape[: -len(dist.cond_shape)] if dist.cond_ndim > 0 else ()
            )
        else:
            batch_shape = ()

        self.dist = dist.merge_transforms()  # Ensure base distribution not transformed
        self._condition = condition
        self.support = constraints.real
        super().__init__(batch_shape=batch_shape, event_shape=dist.shape)

    def sample(self, key, sample_shape=...):
        """Sample the distribution."""
        return self.dist.sample(key, sample_shape, self.condition)

    def sample_with_intermediates(self, key, sample_shape=...):
        """Sample the distribution returning the base distribution sample."""
        z = self.dist.base_dist.sample(key, sample_shape, self._base_condition)
        x = _VectorizedBijection(self.dist.bijection).transform(z, self.condition)
        return x, [z]

    def log_prob(self, value, intermediates=None):
        """Compute the log probabilities."""
        if intermediates is None:
            return self.dist.log_prob(value, self.condition)

        z = intermediates[0]
        _, log_det = _VectorizedBijection(
            self.dist.bijection,
        ).transform_and_log_det(z, self.condition)
        return self.dist.base_dist.log_prob(z, self._base_condition) - log_det

    @property
    def condition(self):
        """condition, wrapped with stop gradient to avoid training."""
        return jax.lax.stop_gradient(self._condition)

    @property
    def _base_condition(self):
        return self.condition if self.dist.base_dist.cond_shape else None


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
    params = numpyro.param(name, params)
    return eqx.combine(params, static)
