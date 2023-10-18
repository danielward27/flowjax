"""
Interfacing with numpyro
==========================

``flowjax.experimental.numpyro`` contains utilities to facilitate interfacing with
numpyro. Note these utilities require numpyro to be installed."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

try:
    import numpyro
except ImportError as e:
    e.add_note(
        "Note, in order to interface with numpyro, it must be installed. Please see "
        "https://num.pyro.ai/en/latest/getting_started.html#installation"
    )
    raise

from numpyro.distributions import constraints

from flowjax.bijections import AbstractBijection
from flowjax.distributions import Transformed
from flowjax.utils import _get_ufunc_signature

# TODO list:
#    - How to support of batch dimensions.
#    - Handle conditioning with a batch dimension?
#    - Do I need to support non-transformed distributions?
#    - Allow control of supports and constraints - will applications of transformations
# to apply constraints lead to problems with reparameterisation?


class VectorizedBijection(eqx.Module):
    "Wrap a flowjax bijection to support vectorization."

    def __init__(self, bijection: AbstractBijection):
        """
        Args:
            bijection (Bijection): flowjax bijection to be wrapped.
            domain (constraints.Constraint, optional): Numpyro constraint.
                Defaults to constraints.real.
        """
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
            self.bijection.transform_and_log_det, log_det=True
        )
        return transform_and_log_det(x, condition)

    def vectorize(self, func, log_det=False):
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
    """Convert a Transformed flowjax distribution to a numpyro distribution."""

    def __init__(
        self,
        dist: Transformed,
        condition: ArrayLike | None = None,
        support: constraints.Constraint = constraints.real,
    ):
        condition = dist._argcheck_and_cast_condition(condition)
        if condition is not None:
            batch_shape = (
                condition.shape[: -len(dist.cond_shape)] if dist.cond_ndim > 0 else ()
            )
        else:
            batch_shape = ()

        self.dist = dist.merge_transforms()  # Ensure base distribution not transformed
        self._condition = condition
        self.support = support
        super().__init__(batch_shape=batch_shape, event_shape=dist.shape)

    def sample(self, key, sample_shape=...):
        return self.dist.sample(key, sample_shape, self.condition)

    def sample_with_intermediates(self, key, sample_shape=...):
        z = self.dist.base_dist.sample(key, sample_shape, self.base_condition)
        x = VectorizedBijection(self.dist.bijection).transform(z, self.condition)
        return x, [z]

    def log_prob(self, value, intermediates=None):
        if intermediates is None:
            return self.dist.log_prob(value, self.condition)
        else:
            z = intermediates[0]
            _, log_det = VectorizedBijection(self.dist.bijection).transform_and_log_det(
                z, self.condition
            )
            return self.dist.base_dist.log_prob(z, self.base_condition) - log_det

    @property
    def condition(self):
        return jax.lax.stop_gradient(self._condition)

    @property
    def base_condition(self):
        return self.condition if self.dist.base_dist.cond_shape else None


def register_params(name: str, model: eqx.Module, filter_spec=eqx.is_inexact_array):
    """Register numpyro params for an equinox module. This simply partitions the
    parameters and static components, registers the parameters using numpyro param,
    then recombines them."""
    params, static = eqx.partition(model, filter_spec)
    params = numpyro.param(name, params)
    model = eqx.combine(params, static)
    return model
