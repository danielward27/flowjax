from typing import Callable
import jax.numpy as jnp
from jaxflows.bijections.abc import Bijection
from jax.scipy.stats import norm
from jax import random
import equinox as eqx
import jax
from jaxflows.bijections.coupling import CouplingStack
from jaxflows.bijections.rational_quadratic_spline import RationalQuadraticSpline


class Flow(eqx.Module):
    bijection: Bijection
    target_dim: int
    base_log_prob: Callable
    base_sample: Callable

    def __init__(
        self,
        bijection: Bijection,
        target_dim: int,
        base_log_prob: Callable = None,
        base_sample: Callable = None,
    ):
        """Form a distribution like object using a base distribution and a
        bijection. Bijection must support extra conditioning variables (or the
        bijection can be wrapped in IgnoreCondition).

        Args:
            bijection (Bijection): Bijection mapping from target distribution to
                the base distribution.
            base_log_prob (Callable, optional): log probability in the base
                distribution. Defaults to standard normal.
            base_sample (Callable, optional): sample function with signature
                (key : PRNGKey, n : int). Defaults to standard normal.
        """
        self.bijection = bijection
        self.target_dim = target_dim

        if base_log_prob:
            self.base_log_prob = base_log_prob
            self.base_sample = base_sample
        else:
            self.base_log_prob = lambda x: norm.logpdf(x).sum(axis=1)
            self.base_sample = lambda key, n: random.normal(key, (n, target_dim))

    def log_prob(self, x: jnp.array, condition=None):
        "Evaluate the log probability of the target distribution. Condition must broadcast to x in dimension 0."
        x = jnp.atleast_2d(x)
        if condition is None:
            condition = jnp.zeros((x.shape[0], 0))  # Empty placeholder
        condition = jnp.broadcast_to(condition, (x.shape[0], condition.shape[1]))
        return self._log_prob(x, condition)

    def _log_prob(self, x: jnp.array, condition: jnp.array):
        z, log_abs_det = jax.vmap(self.bijection.transform_and_log_abs_det_jacobian)(
            x, condition
        )
        p_z = self.base_log_prob(z)
        return p_z + log_abs_det

    def sample(self, key: random.PRNGKey, n: int, condition=None):
        "Sample from the target distribution."
        if condition is None:
            condition = jnp.zeros((n, 0))  # Empty placeholder

        z = self.base_sample(key, n)
        x = jax.vmap(self.bijection.inverse)(z, condition)
        return x


class NeuralSplineFlow(Flow):
    def __init__(
        self,
        key: random.PRNGKey,
        target_dim: int,
        condition_dim: int = 0,
        K: int = 8,
        B: int = 5,
        num_layers: int = 5,
        base_log_prob: Callable = None,
        base_sample: Callable = None,
    ):
        """Convenience constructor for neural spline flow (with coupling layers).
        Note that points outside [-B, B] will not be transformed.

        Args:
            key (random.PRNGKey): Random key.
            target_dim (int): Dimension of the target distribution.
            condition_dim (int): Dimension of extra conditioning variables. Defualts to 0.
            K (int, optional): Number of (inner) spline segments. Defaults to 8.
            B (int, optional): Interval to transform [-B, B]. Defaults to 5.
            base_log_prob (Callable, optional): Log probability in base distribution. Defaults to standard normal.
            base_sample (Callable, optional): Sample function in base distribution. Defaults to standard normal.
        """

        bijection = CouplingStack(
            key=key,
            bijection=RationalQuadraticSpline(K=K, B=B),
            D=target_dim,
            num_layers=num_layers,
            condition_dim=condition_dim,
        )

        super().__init__(bijection, target_dim, base_log_prob, base_sample)

