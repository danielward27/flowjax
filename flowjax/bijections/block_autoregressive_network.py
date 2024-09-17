"""Block Neural Autoregressive bijection implementation."""

from collections.abc import Callable
from math import prod
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import random
from jax.nn import softplus
from jaxtyping import PRNGKeyArray

from flowjax import masks
from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.tanh import LeakyTanh
from flowjax.bisection_search import AutoregressiveBisectionInverter
from flowjax.wrappers import Parameterize, WeightNormalization


class _CallableToBijection(AbstractBijection):
    # Wrap a callable e.g. a function or a callable module. We assume the callable acts
    # on scalar values and log_det can be computed in a stable manner with jax.grad.

    fn: Callable
    shape: ClassVar[tuple] = ()
    cond_shape: ClassVar[None] = None

    def __init__(self, fn: Callable):
        if not callable(fn):
            raise TypeError(f"Expected callable, got {type(fn)}.")
        self.fn = fn

    def transform(self, x, condition=None):
        return self.fn(x)

    def transform_and_log_det(self, x, condition=None):
        y, grad = eqx.filter_value_and_grad(self.fn)(x)
        return y, jnp.log(jnp.abs(grad))

    def inverse(self, y, condition=None):
        raise NotImplementedError

    def inverse_and_log_det(self, y, condition=None):
        raise NotImplementedError


class BlockAutoregressiveNetwork(AbstractBijection):
    r"""Block Autoregressive Network (https://arxiv.org/abs/1904.04676).

    Note that in contrast to the original paper which uses tanh activations, by default
    we use :class:`~flowjax.bijections.tanh.LeakyTanh`. This ensures the codomain of the
    activation is the set of real values, which will ensure properly normalized
    densities (see https://github.com/danielward27/flowjax/issues/102).

    Args:
        key: Jax key
        dim: Dimension of the distribution.
        cond_dim: Dimension of conditioning variables. Defaults to None.
        depth: Number of hidden layers in the network.
        block_dim: Block dimension (hidden layer size is `dim*block_dim`).
        activation: Activation function, either a scalar bijection or a callable that
            computes the activation for a scalar value. Note that the activation should
            be bijective to ensure invertibility of the network and in general should
            map real -> real to ensure that when transforming a distribution (either
            with the forward or inverse), the map is defined across the support of
            the base distribution. Defaults to ``LeakyTanh(3)``.
        inverter: Callable that implements the required numerical method to invert the
            ``BlockAutoregressiveNetwork`` bijection. Must have the signature
            ``inverter(bijection, y, condition=None)``. Defaults to
            ``AutoregressiveBisectionInverter``.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    depth: int
    layers: list
    cond_linear: eqx.nn.Linear | None
    block_dim: int
    activation: AbstractBijection
    inverter: Callable

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        dim: int,
        cond_dim: int | None = None,
        depth: int,
        block_dim: int,
        activation: AbstractBijection | Callable | None = None,
        inverter: Callable | None = None,
    ):
        key, subkey = jr.split(key)

        if activation is None:
            activation = LeakyTanh(3)
        elif isinstance(activation, AbstractBijection):
            if activation.shape != () or activation.cond_shape is not None:
                raise ValueError("Bijection must be unconditional with shape ().")
        else:
            activation = _CallableToBijection(activation)

        self.inverter = (
            AutoregressiveBisectionInverter() if inverter is None else inverter
        )

        layers_and_log_jac_fns = []
        if depth == 0:
            layers_and_log_jac_fns.append(
                block_autoregressive_linear(
                    key,
                    n_blocks=dim,
                    block_shape=(1, 1),
                ),
            )
        else:
            keys = random.split(key, depth + 1)

            block_shapes = [
                (block_dim, 1),
                *[(block_dim, block_dim)] * (depth - 1),
                (1, block_dim),
            ]

            for layer_key, block_shape in zip(keys, block_shapes, strict=True):
                layers_and_log_jac_fns.append(
                    block_autoregressive_linear(
                        layer_key,
                        n_blocks=dim,
                        block_shape=block_shape,
                    ),
                )

        if cond_dim is not None:
            layer0_out_dim = layers_and_log_jac_fns[0][0].out_features
            self.cond_linear = eqx.nn.Linear(
                cond_dim, layer0_out_dim, use_bias=False, key=subkey
            )
        else:
            self.cond_linear = None

        self.depth = depth
        self.layers = layers_and_log_jac_fns
        self.block_dim = block_dim
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)
        self.activation = activation

    def transform(self, x, condition=None):
        for i, (layer, _) in enumerate(self.layers[:-1]):
            x = layer(x)
            if i == 0 and condition is not None:
                assert self.cond_linear is not None
                x += self.cond_linear(condition)
            x = eqx.filter_vmap(self.activation.transform)(x)
        return self.layers[-1][0](x)

    def transform_and_log_det(self, x, condition=None):
        log_dets_3ds = []
        for i, (linear, log_jacobian_fn) in enumerate(self.layers[:-1]):
            x = linear(x)
            if i == 0 and condition is not None:
                assert self.cond_linear is not None
                x += self.cond_linear(condition)
            log_dets_3ds.append(log_jacobian_fn(linear))
            x, log_det_3d = self._activation_and_log_jacobian_3d(x)
            log_dets_3ds.append(log_det_3d)

        linear, log_jacobian_fn = self.layers[-1]
        x = linear(x)
        log_dets_3ds.append(log_jacobian_fn(linear))

        log_det = log_dets_3ds[-1]
        for log_jacobian in reversed(log_dets_3ds[:-1]):
            log_det = logmatmulexp(log_det, log_jacobian)
        return x, log_det.sum()

    def inverse(self, y, condition=None):
        return self.inverter(self, y, condition)

    def inverse_and_log_det(self, y, condition=None):
        x = self.inverter(self, y, condition)
        _, forward_log_det = self.transform_and_log_det(x, condition)
        return x, -forward_log_det

    def _activation_and_log_jacobian_3d(self, x):
        """Compute activation and the log determinant (blocks, block_dim, block_dim)."""
        x, log_abs_grads = eqx.filter_vmap(self.activation.transform_and_log_det)(x)
        log_det_3d = jnp.full((self.shape[0], self.block_dim, self.block_dim), -jnp.inf)
        diag_idxs = jnp.arange(self.block_dim)
        log_det_3d = log_det_3d.at[:, diag_idxs, diag_idxs].set(
            log_abs_grads.reshape(self.shape[0], self.block_dim)
        )
        return x, log_det_3d


def block_autoregressive_linear(
    key: PRNGKeyArray,
    *,
    n_blocks: int,
    block_shape: tuple,
) -> tuple[eqx.nn.Linear, Callable]:
    """Block autoregressive linear layer (https://arxiv.org/abs/1904.04676).

    Returns:
        Tuple containing 1) an equinox.nn.Linear layer, with the weights wrapped
        in order to provide a block lower triangular mask and weight normalisation.
        2) callable taking the linear layer, returning the log block diagonal weights.

    Args:
        key: Random key.
        n_blocks: Number of diagonal blocks (dimension of original input).
        block_shape: The shape of the blocks.
    """
    out_features, in_features = (b * n_blocks for b in block_shape)
    linear = eqx.nn.Linear(in_features, out_features, key=key)
    block_diag_mask = masks.block_diag_mask(block_shape, n_blocks)
    block_tril_mask = masks.block_tril_mask(block_shape, n_blocks)

    def apply_mask(weight):
        weight = jnp.where(block_tril_mask, weight, 0)
        return jnp.where(block_diag_mask, softplus(weight), weight)

    weight = WeightNormalization(Parameterize(apply_mask, linear.weight))
    linear = eqx.tree_at(lambda linear: linear.weight, linear, replace=weight)

    def linear_to_log_block_diagonal(linear: eqx.nn.Linear):
        idxs = jnp.where(block_diag_mask, size=prod(block_shape) * n_blocks)
        jac_3d = linear.weight[idxs].reshape(n_blocks, *block_shape)
        return jnp.log(jac_3d)

    return linear, linear_to_log_block_diagonal


def logmatmulexp(x, y):
    """Numerically stable version of ``(x.log() @ y.log()).exp()``.

    From numpyro https://github.com/pyro-ppl/numpyro/blob/
    f2ff89a3a7147617e185eb51148eb15d56d44661/numpyro/distributions/util.py#L387.
    """
    x_shift = jax.lax.stop_gradient(jnp.amax(x, -1, keepdims=True))
    y_shift = jax.lax.stop_gradient(jnp.amax(y, -2, keepdims=True))
    xy = jnp.log(jnp.matmul(jnp.exp(x - x_shift), jnp.exp(y - y_shift)))
    return xy + x_shift + y_shift
