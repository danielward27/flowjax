from flowjax.bijections.abc import ParameterisedBijection
import jax.numpy as jnp
import jax
from jax.scipy.special import erfc, erfinv
from functools import partial

class ExtremeValueActivation(ParameterisedBijection):
    """
    ExtremeValueActivation (D. Prangle, T. Hickling)

    This transform is Reals -> Reals.
    """
    def __init__( self, min_tail_param=1e-3, max_tail_param=1):
        self.min_tail_param = min_tail_param
        self.max_tail_param = max_tail_param

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def transform(self, x, pos_tail, neg_tail):
        """
        x in reals
        """
        x = jnp.clip(x, a_min=-1_000, a_max=1_000)
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)

        g = jnp.power(1 + tail_param * jnp.abs(x), -1 / tail_param)

        transformed = sign * jnp.sqrt(2) * erfinv(1 - g)

        return transformed

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def transform_and_log_abs_det_jacobian(self, x, pos_tail, neg_tail):
        x = jnp.clip(x, a_min=-1_000, a_max=1_000)
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)

        inner = 1 + tail_param * jnp.abs(x)
        g = jnp.power(inner, -1 / tail_param)

        transformed = sign * jnp.sqrt(2) * erfinv(1 - g)
        

        dt_dx = jnp.power(inner, -1 - 1/tail_param)
        dt_dx *= 0.5 * sign * sign * jnp.sqrt(2) * jnp.sqrt(jnp.pi)
        dt_dx *= jnp.exp(jnp.square(erfinv(1 - jnp.power(inner, -1/tail_param))))

        logabsdet = jnp.log(dt_dx)

        return transformed, logabsdet

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def inverse(self, u, pos_tail, neg_tail):
        """
        From reals 
        """
        sign = jnp.sign(u)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)
        g = erfc(jnp.abs(u) / jnp.sqrt(2))

        transformed = sign / tail_param
        transformed *= jnp.power(g, -tail_param) - 1
        return transformed

    def num_params(self, dim: int) -> int:
        return dim * 2
    
    def get_ranks(self, dim: int):
        return jnp.repeat(jnp.arange(dim), 2)

    def get_args(self, params):
        params = params.reshape((-1, 2))
        tail_params = jax.nn.sigmoid(params)
        tail_params *= self.max_tail_param - self.min_tail_param
        tail_params += self.min_tail_param
        return tail_params[:, 0], tail_params[:, 1]

class TailTransformation(ParameterisedBijection):
    """
    TailTransformation (D. Prangle, T. Hickling)

    This transform is (-1, 1) -> Reals.
    """
    def __init__( self, min_tail_param=1e-3, max_tail_param=1):
        self.min_tail_param = min_tail_param
        self.max_tail_param = max_tail_param

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def transform(self, x, pos_tail, neg_tail):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)
        neg_tail_inv = -1 / tail_param

        transformed = jnp.power(1 + sign * tail_param * x, neg_tail_inv)
        transformed = sign * (1 - transformed)

        return transformed

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def transform_and_log_abs_det_jacobian(self, x, pos_tail, neg_tail):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)
        neg_tail_inv = -1 / tail_param

        transformed = jnp.power(1 + sign * tail_param * x, neg_tail_inv)
        transformed = sign * (1 - transformed)

        dt_dx = jnp.power(1 + sign * tail_param * x, neg_tail_inv - 1)
        logabsdet = jnp.log(dt_dx)

        return transformed, logabsdet

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def inverse(self, u, pos_tail, neg_tail):
        sign = jnp.sign(u)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)

        transformed = sign / tail_param
        transformed *= jnp.power(1 - jnp.abs(u), -tail_param) - 1
        
        return transformed

    def num_params(self, dim: int) -> int:
        return dim * 2
    
    def get_ranks(self, dim: int):
        return jnp.repeat(jnp.arange(dim), 2)

    def get_args(self, params):
        params = params.reshape((-1, 2))
        tail_params = jax.nn.sigmoid(params)
        tail_params *= self.max_tail_param - self.min_tail_param
        tail_params += self.min_tail_param
        return tail_params[:, 0], tail_params[:, 1]