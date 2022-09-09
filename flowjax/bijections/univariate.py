from flowjax.bijections.abc import ParameterisedBijection
import equinox as eqx
import jax.numpy as jnp
from flowjax.bijections.abc import Bijection
from jax.random import KeyArray
from flowjax.utils import Array
from functools import partial
import jax

class Univariate(Bijection):
    """
    A class implementing a Bijection for the univariate case. 
    """
    d: int
    D: int
    bijection: ParameterisedBijection
    conditioner: eqx.nn.MLP
    cond_dim: int

    def __init__(
        self,
        key: KeyArray,
        bijection: ParameterisedBijection,
        cond_dim: int = 0,
        nn_width: int = 1, # NN params won't matter unless there is a cond dim
        nn_depth: int = 1,
    ):
        self.bijection = bijection
        self.d = 0 # Input dim is always 1 for univariate case
        self.D = 1 # Output always 1
        self.cond_dim = cond_dim

        self.conditioner = eqx.nn.MLP(
            in_size=1 + cond_dim,
            out_size=bijection.num_params(self.D),
            width_size=nn_width,
            depth=nn_depth,
            key=key,
        )

    def get_params(self, condition=None):
        condition = jnp.empty((0,)) if condition is None else condition
        return jax.vmap(self._get_params, [0])(condition)

    def _get_params(self, condition=None):
        x_cond = jnp.array([1]) # Constant input
        if condition is not None:
            cond = jnp.concatenate((x_cond, condition))
        else:
            cond = x_cond

        return self.conditioner(cond)

    def transform(self, x: Array, condition=None):
        assert x.ndim == 1, 'Attempting univariate transform on multidimensional input!'
        bijection_params = self._get_params(condition)
        bijection_args = self.bijection.get_args(bijection_params)
        return self.bijection.transform(x, *bijection_args)

    def transform_and_log_abs_det_jacobian(self, x: Array, condition=None):
        assert x.ndim == 1, 'Attempting univariate transform on multidimensional input!'
        bijection_params = self._get_params(condition)
        bijection_args = self.bijection.get_args(bijection_params)
        y_trans, log_abs_det = self.bijection.transform_and_log_abs_det_jacobian(
            x, *bijection_args
        )
        return y_trans, log_abs_det

    def inverse(self, y: Array, condition=None):
        condition = jnp.array([]) if condition is None else condition
        x_cond, y_trans = y[: self.d], y[self.d :]
        cond = jnp.concatenate((x_cond, condition))
        bijection_params = self.conditioner(cond)
        bijection_args = self.bijection.get_args(bijection_params)
        x_trans = self.bijection.inverse(y_trans, *bijection_args)
        x = jnp.concatenate((x_cond, x_trans))
        return x