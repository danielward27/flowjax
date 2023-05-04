"""Block autoregressive neural network components."""
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, random
from jax.nn.initializers import glorot_uniform
from jax.random import KeyArray
from flowjax.bijections.tanh import Tanh
from flowjax.masks import block_diag_mask, block_tril_mask


def _tanh_log_grad(x):
    """log gradient vector of tanh transformation."""
    return -2 * (x + jax.nn.softplus(-2 * x) - jnp.log(2.0))


class BlockAutoregressiveLinear(eqx.Module):
    """Block autoregressive neural network layer (https://arxiv.org/abs/1904.04676).
    Conditioning variables are incorporated by appending columns (one for each
    conditioning variable) to the left of the block diagonal weight matrix.
    """

    n_blocks: int
    block_shape: tuple
    cond_dim: int | None
    weights: Array
    bias: Array
    weight_log_scale: Array
    in_features: int
    out_features: int
    b_diag_mask: Array
    b_diag_mask_idxs: Array
    b_tril_mask: Array

    def __init__(
        self,
        key: KeyArray,
        n_blocks: int,
        block_shape: tuple,
        cond_dim: int | None = None,
        init: Callable = glorot_uniform(),
    ):
        """
        Args:
            key KeyArray: Random key
            n_blocks (int): Number of diagonal blocks (dimension of original input).
            block_shape (tuple): The shape of the (unconstrained) blocks.
            cond_dim (int | None): Number of additional conditioning variables.
                Defaults to None.
            init (Callable): Default initialisation method for the weight
                matrix. Defaults to ``glorot_uniform()``.
        """
        self.cond_dim = cond_dim

        if cond_dim is None:
            cond_dim = 0

        cond_size = (block_shape[0] * n_blocks, cond_dim)

        self.b_diag_mask = jnp.column_stack(
            (block_diag_mask(block_shape, n_blocks), jnp.zeros(cond_size, jnp.int32))
        )

        self.b_tril_mask = jnp.column_stack(
            (block_tril_mask(block_shape, n_blocks), jnp.ones(cond_size, jnp.int32))
        )

        self.b_diag_mask_idxs = jnp.where(
            self.b_diag_mask, size=block_shape[0] * block_shape[1] * n_blocks
        )  # type: ignore

        in_features, out_features = (
            block_shape[1] * n_blocks + cond_dim,
            block_shape[0] * n_blocks,
        )

        *w_key, bias_key, scale_key = random.split(key, n_blocks + 2)

        self.weights = init(w_key[0], (out_features, in_features)) * (
            self.b_tril_mask + self.b_diag_mask
        )
        self.bias = (random.uniform(bias_key, (out_features,)) - 0.5) * (
            2 / jnp.sqrt(out_features)
        )

        self.n_blocks = n_blocks
        self.block_shape = block_shape
        self.cond_dim = cond_dim
        self.weight_log_scale = jnp.log(random.uniform(scale_key, (out_features, 1)))
        self.in_features = in_features
        self.out_features = out_features

    def get_normalised_weights(self):
        """Carries out weight normalisation."""
        weights = (
            jnp.exp(self.weights) * self.b_diag_mask + self.weights * self.b_tril_mask
        )
        weight_norms = jnp.linalg.norm(weights, axis=-1, keepdims=True)
        return jnp.exp(self.weight_log_scale) * weights / weight_norms

    def __call__(self, x, condition=None):
        """returns output y, and components of weight matrix needed log_det component
        (n_blocks, block_shape[0], block_shape[1]).
        """
        weights = self.get_normalised_weights()
        if condition is not None:
            x = jnp.concatenate((x, condition))
        y = weights @ x + self.bias
        jac_3d = weights[self.b_diag_mask_idxs].reshape(
            self.n_blocks, *self.block_shape
        )
        return y, jnp.log(jac_3d)


def _block_tanh_activation(n_blocks: int):
    """Returms a Tanh activation compatible with a block neural autoregressive network,
    i.e. returning the transformed variable and the absolute log gradients as a 3D
    array with shape (n_blocks,) + block_shape
    """

    def activation(x: Array):
        tanh = Tanh()
        y, log_jacobian = jax.vmap(tanh.transform_and_log_det)(x)
        return y, _reshape_jacobian_to_3d(log_jacobian, n_blocks)

    return activation


def _reshape_jacobian_to_3d(vals, n_blocks):
    block_dim = vals.shape[0] // n_blocks
    log_det = jnp.full((n_blocks, block_dim, block_dim), -jnp.inf)
    log_det = log_det.at[:, jnp.arange(block_dim), jnp.arange(block_dim)].set(
        vals.reshape(n_blocks, block_dim)
    )
    return log_det
