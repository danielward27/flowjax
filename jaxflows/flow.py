from typing import Callable
import jax.numpy as jnp
from jaxflows.bijections.abc import Bijection
from jax.scipy.stats import norm
from jax import random
import equinox as eqx
import jax


class Flow(eqx.Module):
    bijection: Bijection
    target_dim: int
    base_log_prob: Callable
    base_sample: Callable

    def __init__(
        self,
        bijection: Bijection,
        target_dim: int,
        condition_dim: int = 0,  # TODO do we want to have this
        base_log_prob: Callable = None,
        base_sample: Callable = None,
    ):
        """Form a distribution like object using a base distribution and a
        bijection.

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
        "Evaluate the log probability of the target distribution."
        if condition is None:
            condition = jnp.zeros((x.shape[0], 0))  # Empty placeholder
        return self._log_prob(x, condition)

    @eqx.filter_jit
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
