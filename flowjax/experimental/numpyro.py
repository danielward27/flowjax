"""
Interfacing with numpyro
==========================

Note these utilities require `numpyro <https://github.com/pyro-ppl/numpyro>`_ to be
installed. Supporting complex inference approaches such as MCMC or variational inference
with arbitrary probabilistic models is out of the scope of this package. However, we do
provide an (experimental) wrapper class, :class:`TransformedToNumpyro`, which will wrap
a flowjax :class:`~flowjax.distributions.Transformed` distribution, into a 
`numpyro <https://github.com/pyro-ppl/numpyro>`_ distribution.
This can be used for example to embed normalising flows into arbitrary
probabilistic models. Here is a simple example

    .. doctest::

        >>> from numpyro.infer import MCMC, NUTS
        >>> from flowjax.experimental.numpyro import TransformedToNumpyro
        >>> from numpyro import sample
        >>> from flowjax.distributions import Normal
        >>> import jax.random as jr
        >>> import numpy as np

        >>> def numpyro_model(X, y):
        ...     "Example regression model defined in terms of flowjax distributions"
        ...     beta = sample("beta", TransformedToNumpyro(Normal(np.zeros(2))))
        ...     sample("y", TransformedToNumpyro(Normal(X @ beta)), obs=y)

        >>> X = np.random.randn(100, 2)
        >>> beta_true = np.array([-1, 1])
        >>> y = X @ beta_true + np.random.randn(100)
        >>> mcmc = MCMC(NUTS(numpyro_model), num_warmup=10, num_samples=100)
        >>> mcmc.run(jr.PRNGKey(0), X, y)


"""

from typing import Any, Callable

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

from flowjax.bijections import Bijection
from flowjax.distributions import Transformed
from flowjax.utils import _get_ufunc_signature

PyTree = Any
# TODO list:
#    - How to add support of batch dimensions.
#    - Do I need to support non-transformed distributions?
#    - Allow control of supports and constraints - will applications of transformations
#           to apply constraints lead to problems with reparameterisation?


class _VectorizedBijection:
    "Wrap a flowjax bijection to support vectorization."

    def __init__(self, bijection: Bijection):
        """
        Args:
            bijection (AbstractBijection): flowjax bijection to be wrapped.
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
    """Convert a :class:`Transformed` flowjax distribution to a numpyro distribution. We
    assume the support of the distribution is unbounded.
    """

    def __init__(
        self,
        dist: Transformed,
        condition: ArrayLike | None = None,
    ):
        """
        Args:
            dist (Transformed): The distribution
            condition (ArrayLike | None, optional): Conditioning variables. Any
                leading batch dimensions will be converted to a batch dimension in
                the numpyro distribution. Defaults to None.
        """
        condition = dist._argcheck_and_cast_condition(condition)
        if condition is not None:
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
        return self.dist.sample(key, sample_shape, self.condition)

    def sample_with_intermediates(self, key, sample_shape=...):
        z = self.dist.base_dist.sample(key, sample_shape, self.base_condition)
        x = _VectorizedBijection(self.dist.bijection).transform(z, self.condition)
        return x, [z]

    def log_prob(self, value, intermediates=None):
        if intermediates is None:
            return self.dist.log_prob(value, self.condition)
        else:
            z = intermediates[0]
            _, log_det = _VectorizedBijection(
                self.dist.bijection
            ).transform_and_log_det(z, self.condition)
            return self.dist.base_dist.log_prob(z, self.base_condition) - log_det

    @property
    def condition(self):
        return jax.lax.stop_gradient(self._condition)

    @property
    def base_condition(self):
        return self.condition if self.dist.base_dist.cond_shape else None


def register_params(
    name: str, model: PyTree, filter_spec: Callable | PyTree = eqx.is_inexact_array
):
    """Register numpyro params for an arbitrary pytree (e.g. an equinox module,
    flowjax distribution/bijection). This simply partitions the parameters and static
    components, registers the parameters using numpyro.param, then recombines them.
    This should be called from within an inference context, e.g. within a numpyro
    model or guide function to have an effect.

    Args:
        name (str): Name for the parameter set.
        model (PyTree): The pytree (e.g. an equinox module, flowjax distribution,
            or a flowjax bijection).
        filter_spec (Callable | PyTree): Equinox `filter_spec` for specifying trainable
            parameters. Either a callable `leaf -> bool`, or a PyTree with prefix
            structure matching `dist` with True/False values. Defaults to
            `eqx.is_inexact_array`.

    """
    params, static = eqx.partition(model, filter_spec)
    params = numpyro.param(name, params)
    model = eqx.combine(params, static)
    return model
