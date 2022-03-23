from typing import Callable
import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.bijections.abc import Bijection
from jax import random
from math import prod
from jax.nn.initializers import glorot_uniform
from jax import lax


def b_diag_mask(block_shape: tuple, num_blocks: int):
    return jax.scipy.linalg.block_diag(
        *[jnp.ones(block_shape, int) for _ in range(num_blocks)]
    )


def b_tril_mask(block_shape: tuple, num_blocks: int):
    "Upper triangular block mask, excluding diagonal blocks."
    mask = jnp.zeros((block_shape[0] * num_blocks, block_shape[1] * num_blocks))

    for i in range(num_blocks):
        mask = mask.at[
            (i + 1) * block_shape[0] :, i * block_shape[1] : (i + 1) * block_shape[1]
        ].set(1)
    return mask


class BlockAutoregressiveLinear(eqx.Module):
    num_blocks: int
    block_shape: tuple
    W: jnp.ndarray
    bias: jnp.ndarray
    W_log_scale: jnp.ndarray
    in_features: int
    out_features: int
    _b_diag_mask: jnp.ndarray
    _b_diag_mask_idxs: jnp.ndarray
    _b_tril_mask: jnp.ndarray

    def __init__(
        self,
        key: random.PRNGKey,
        num_blocks: int,
        block_shape: tuple,
        init=glorot_uniform(),
    ):
        self.block_shape = block_shape
        self.num_blocks = num_blocks

        self._b_diag_mask = b_diag_mask(block_shape, num_blocks)
        self._b_diag_mask_idxs = jnp.where(self._b_diag_mask)
        self._b_tril_mask = b_tril_mask(block_shape, num_blocks)

        in_features, out_features = (
            block_shape[1] * num_blocks,
            block_shape[0] * num_blocks,
        )

        *w_key, bias_key, scale_key = random.split(key, num_blocks + 2)

        self.W = init(w_key[0], (out_features, in_features)) * (
            self.b_tril_mask + self.b_diag_mask
        )
        self.bias = (random.uniform(bias_key, (out_features,)) - 0.5) * (
            2 / jnp.sqrt(out_features)
        )
        self.W_log_scale = jnp.log(random.uniform(scale_key, (out_features, 1)))
        self.in_features = in_features
        self.out_features = out_features

    def __call__(
        self, x
    ):  # TODO once trained the jacobian is fixed. Maybe this can be exploited?
        "returns output y, and components of weight matrix needed log_det component (n_blocks, block_shape[0], block_shape[1])"
        W = jnp.exp(self.W) * self.b_diag_mask + self.W * self.b_tril_mask
        W_norms = jnp.linalg.norm(W, axis=-1, keepdims=True)

        W = jnp.exp(self.W_log_scale) * W / W_norms  # Weight normalisation
        y = W @ x + self.bias
        log_jac = self.W_log_scale + self.W - jnp.log(W_norms)
        log_jac_3d = log_jac[self.b_diag_mask_idxs].reshape(
            self.num_blocks, *self.block_shape
        )
        return y, log_jac_3d

    @property
    def b_diag_mask(self):
        return jax.lax.stop_gradient(self._b_diag_mask)

    @property
    def b_diag_mask_idxs(self):
        return jax.lax.stop_gradient(self._b_diag_mask_idxs)

    @property
    def b_tril_mask(self):
        return jax.lax.stop_gradient(self._b_tril_mask)


def logmatmulexp(
    x, y
):  # TODO consider version like paper eq. 11 or are they equivilent?
    """
    Numerically stable version of ``(x.log() @ y.log()).exp()``. From numpyro https://github.com/pyro-ppl/numpyro/blob/f2ff89a3a7147617e185eb51148eb15d56d44661/numpyro/distributions/util.py#L387
    """
    x_shift = lax.stop_gradient(jnp.amax(x, -1, keepdims=True))
    y_shift = lax.stop_gradient(jnp.amax(y, -2, keepdims=True))
    xy = jnp.log(jnp.matmul(jnp.exp(x - x_shift), jnp.exp(y - y_shift)))
    return xy + x_shift + y_shift


class _TanhBNAF:
    """
    Tanh transformation compatible with BNAF (log_abs_det provided as 3D array).
    Condition is ignored. Output shape is (num_blocks, *block_size), where
    output[i] is the log jacobian for the iith block.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks

    def __call__(self, x, condition=jnp.array([])):
        d = x.shape[0] // self.num_blocks
        log_det_vals = -2 * (x + jax.nn.softplus(-2 * x) - jnp.log(2.0))
        log_det = jnp.full((self.num_blocks, d, d), -jnp.inf)
        log_det = log_det.at[:, jnp.arange(d), jnp.arange(d)].set(
            log_det_vals.reshape(self.num_blocks, d)
        )
        return jnp.tanh(x), log_det


class BlockAutoregressiveNetwork(eqx.Module, Bijection):
    n_layers: int
    layers: list
    activation: Callable

    def __init__(
        self,
        key: random.PRNGKey,
        dim: int,
        n_layers: int = 3,
        block_size: tuple = (8, 8),
        activation=_TanhBNAF,
    ):

        self.n_layers = n_layers

        layers = []

        block_sizes = [
            (block_size[0], 1),
            *[block_size] * (n_layers - 2),
            (1, block_size[1]),
        ]
        for size in block_sizes:
            key, subkey = random.split(key)
            layers.extend(
                [BlockAutoregressiveLinear(subkey, dim, size), activation(dim)]
            )
        self.layers = layers[:-1]
        self.activation = activation

    def transform(self, x: jnp.ndarray, condition: jnp.ndarray = jnp.array([])):
        y = x
        for layer in self.layers:
            y = layer(y)[0]
        return y

    def transform_and_log_abs_det_jacobian(self, x, condition=jnp.array([])):
        y = x
        log_det_3ds = []

        for layer in self.layers:
            y, log_det_3d = layer(y)
            log_det_3ds.append(log_det_3d)

        logdet = log_det_3ds[-1]
        for ld in reversed(log_det_3ds[:-1]):
            logdet = logmatmulexp(logdet, ld)
        return y, logdet.sum()

    def inverse():
        return NotImplementedError(
            """
        This transform would require numerical methods for inversion..
        """
        )

