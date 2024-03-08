"""Block Neural Autoregressive bijection implementation."""

from collections.abc import Callable
from math import prod
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, random

from flowjax import masks
from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.softplus import SoftPlus
from flowjax.bijections.tanh import LeakyTanh
from flowjax.bisection_search import AutoregressiveBisectionInverter
from flowjax.wrappers import BijectionReparam, WeightNormalization, Where


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
    activation is the set of real values, which will ensure properly normalised
    densities (see https://github.com/danielward27/flowjax/issues/102).

    Args:
        key: Jax PRNGKey
        dim: Dimension of the distribution.
        cond_dim: Dimension of conditioning variables.
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
    block_dim: int
    activation: AbstractBijection
    inverter: Callable

    def __init__(
        self,
        key: Array,
        *,
        dim: int,
        cond_dim: int | None,
        depth: int,
        block_dim: int,
        activation: AbstractBijection | Callable | None = None,
        inverter: Callable | None = None,
    ):
        cond_dim = 0 if cond_dim is None else cond_dim
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
                    extra_inputs=cond_dim,
                ),
            )
        else:
            keys = random.split(key, depth + 1)

            block_shapes = [
                (block_dim, 1),
                *[(block_dim, block_dim)] * (depth - 1),
                (1, block_dim),
            ]
            cond_dims = [cond_dim] + [0] * depth

            for layer_key, block_shape, cond_d in zip(
                keys,
                block_shapes,
                cond_dims,
                strict=True,
            ):
                layers_and_log_jac_fns.append(
                    block_autoregressive_linear(
                        layer_key,
                        n_blocks=dim,
                        block_shape=block_shape,
                        extra_inputs=cond_d,
                    ),
                )

        self.depth = depth
        self.layers = layers_and_log_jac_fns
        self.block_dim = block_dim
        self.shape = (dim,)
        self.cond_shape = (cond_dim,) if cond_dim != 0 else None
        self.activation = activation

    def transform(self, x, condition=None):
        x = x if condition is None else jnp.concatenate((x, condition))
        for layer, _ in self.layers[:-1]:
            x = layer(x)
            x = eqx.filter_vmap(self.activation.transform)(x)
        return self.layers[-1][0](x)

    def transform_and_log_det(self, x, condition=None):
        x = x if condition is None else jnp.concatenate((x, condition))
        log_dets_3ds = []
        for linear, log_jacobian_fn in self.layers[:-1]:
            x = linear(x)
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
        log_det_3d = log_det_3d.at[
            :,
            jnp.arange(self.block_dim),
            jnp.arange(self.block_dim),
        ].set(log_abs_grads.reshape(self.shape[0], self.block_dim))
        return x, log_det_3d


def block_autoregressive_linear(
    key: Array,
    *,
    n_blocks: int,
    block_shape: tuple,
    extra_inputs: int = 0,
) -> tuple[eqx.nn.Linear, Array]:
    """Block autoregressive linear layer (https://arxiv.org/abs/1904.04676).

    Returns:
        Tuple containing 1) an equinox.nn.Linear layer, with the weights wrapped
        in order to provide a block lower triangular mask and weight normalisation.
        2) callable taking the linear layer, returning the log block diagonal weights.

    Args:
        key: Random key.
        n_blocks: Number of diagonal blocks (dimension of original input).
        block_shape: The shape of the blocks.
        extra_inputs: Number of additional input variables for which masking is not
        required. Defaults to 0.
    """
    in_features = block_shape[1] * n_blocks + extra_inputs
    out_features = block_shape[0] * n_blocks
    linear = eqx.nn.Linear(in_features, out_features, key=key)

    def _right_pad(arr, val, width):
        return jnp.column_stack((arr, jnp.full((arr.shape[0], width), val, int)))

    block_diag_mask = _right_pad(
        masks.block_diag_mask(block_shape, n_blocks),
        val=False,
        width=extra_inputs,
    )

    block_tril_mask = _right_pad(
        masks.block_tril_mask(block_shape, n_blocks),
        val=True,
        width=extra_inputs,
    )

    weight = Where(block_tril_mask, linear.weight, 0)
    weight = Where(
        block_diag_mask,
        BijectionReparam(weight, SoftPlus(), invert_on_init=False),
        weight,
    )
    weight = WeightNormalization(weight)
    linear = eqx.tree_at(lambda linear: linear.weight, linear, replace=weight)

    def layer_to_log_jacobian_3d(linear: eqx.nn.Linear):
        idxs = jnp.where(block_diag_mask, size=prod(block_shape) * n_blocks)
        jac_3d = linear.weight[idxs].reshape(n_blocks, *block_shape)
        return jnp.log(jac_3d)

    return linear, layer_to_log_jacobian_3d


def logmatmulexp(x, y):
    """Numerically stable version of ``(x.log() @ y.log()).exp()``.

    From numpyro https://github.com/pyro-ppl/numpyro/blob/
    f2ff89a3a7147617e185eb51148eb15d56d44661/numpyro/distributions/util.py#L387.
    """
    x_shift = jax.lax.stop_gradient(jnp.amax(x, -1, keepdims=True))
    y_shift = jax.lax.stop_gradient(jnp.amax(y, -2, keepdims=True))
    xy = jnp.log(jnp.matmul(jnp.exp(x - x_shift), jnp.exp(y - y_shift)))
    return xy + x_shift + y_shift
