from typing import Callable
import jax.numpy as jnp
from jaxflows.bijections.abc import Bijection
from jax.scipy.stats import norm
from jax import random
import equinox as eqx
import jax

class Flow(eqx.Module):
    bijection : Bijection
    target_dim : int
    base_log_prob : Callable
    base_sample : Callable

    def __init__(
        self,
        bijection : Bijection,
        target_dim : int,
        base_log_prob : Callable = None,
        base_sample : Callable = None):
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
            
    def log_prob(self, x : jnp.array):
        "Evaluate the log probability of the target distribution."
        z, log_abs_det = jax.vmap(self.bijection.transform_and_log_abs_det_jacobian)(x)
        p_z = self.base_log_prob(z)
        return p_z + log_abs_det

    def sample(self, key : random.PRNGKey, n : int):
        "Sample from the target distribution."
        z = self.base_sample(key, n)
        x = jax.vmap(self.bijection.inverse)(z)
        return x
        
