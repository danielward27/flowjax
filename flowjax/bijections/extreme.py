from flowjax.bijections.abc import ParameterisedBijection
import jax.numpy as jnp
import jax
from functools import partial

class ExtremeValueActivation(ParameterisedBijection):
    """
    ExtremeValueActivation (D. Prangle, T. Hickling)
    
    Args:
        K (int): Number of inner knots B: (int):
        B: (int): Interval to transform [-B, B]
    """
    def __init__(
        self,
        tail_constraint=1,
    ):
        self.tail_constraint = tail_constraint

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def transform(self, x, pos_tail, neg_tail):
        pass
    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def inverse(self, u, pos_tail, neg_tail):
        pass


class TailTransformation(ParameterisedBijection):
    """
    TailTransformation (D. Prangle, T. Hickling)

    This transform is (-1, 1) -> Reals.
    As such 
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