import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.bijections.abc import Bijection
from jax import random
from math import prod
from jax.nn.initializers import glorot_uniform


class BlockAutoregressiveLinear(eqx.Module, Bijection):
    num_blocks: int
    block_shape: tuple
    block_diag_mask: jnp.ndarray  # TODO Do I need to buffer these?
    block_tril_mask: jnp.ndarray
    W: jnp.ndarray
    bias: jnp.ndarray

    def __init__(
        self,
        key: random.PRNGKey,
        num_blocks: int,
        block_shape: tuple,
        init=glorot_uniform(),
    ):
        self.block_shape = block_shape
        self.num_blocks = num_blocks

        block_shapes = [
            (block_shape[0], 1),
            *[block_shape for _ in range(num_blocks - 2)],
            (1, block_shape[1]),
        ]

        self.block_diag_mask = jax.scipy.linalg.block_diag(
            *[jnp.ones(s, int) for s in block_shapes]
        )
        self.block_tril_mask = block_tril_mask(block_shapes)
        w_key, bias_key = random.split(key)

        w_shape = [prod(a) for a in zip(*block_shapes)]
        self.W = init(w_key, w_shape)*(self.block_tril_mask + self.block_diag_mask)
        self.bias = init(bias_key, (w_shape[1],))

    def transform(self, x):
        w = jnp.softplus(self.W) * self.block_diag_mask + self.W * self.block_tril_mask
        return w @ x + self.bias

    def transform_and_log_abs_det_jacobian(self, x, *args, condition=...):
        w = jnp.exp(self.W) * self.block_diag_mask + self.W * self.block_tril_mask
        y = w @ x + self.bias

        log_det = self.W[self.block_diag_mask.astype(bool)]


        log_det = 0
        for i in self.num_blocks:
            log_det_i = 
            log_det.append(log_det_i)

        dense_log_det = self.W * self.block_diag_mask
        jax.nn.logsumexp()

        # Is numpyro missing an exp? logmatmul exp and already reparamed using exp?

        log_abs_det = 1  # TODO calculate log_abs_det here
        return y, log_abs_det

    # TODO Maybe try bnaf log_abs_det and compare to true using jacobian function

    def inverse():
        NotImplementedError("Inverse is not implemented for this bijection.")


def block_tril_mask(block_shapes: list):
    "Lower triangular block mask, excluding diagonal blocks."
    arrays = []
    for i, shape1 in enumerate(block_shapes):
        row = []
        for j, shape2 in enumerate(block_shapes):  # n_rows from
            if j < i:
                row.append(jnp.ones((shape1[0], shape2[1])))
            else:
                row.append(jnp.zeros((shape1[0], shape2[1])))
        arrays.append(row)
    return jnp.block(arrays)


