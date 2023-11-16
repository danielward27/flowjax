"""Block Neural Autoregressive bijection implementation."""
from collections.abc import Callable
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.random import KeyArray

from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.tanh import LeakyTanh
from flowjax.nn.block_autoregressive import BlockAutoregressiveLinear


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
    """
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    depth: int
    layers: list
    block_dim: int
    activation: AbstractBijection

    def __init__(
        self,
        key: KeyArray,
        *,
        dim: int,
        cond_dim: int | None,
        depth: int,
        block_dim: int,
        activation: AbstractBijection | Callable | None = None,
    ):
        if activation is None:
            activation = LeakyTanh(3)
        elif isinstance(activation, AbstractBijection):
            if activation.shape != () or activation.cond_shape is not None:
                raise ValueError("Bijection must be unconditional with shape ().")
        else:
            activation = _CallableToBijection(activation)

        layers = []
        if depth == 0:
            layers.append(
                BlockAutoregressiveLinear(
                    key,
                    n_blocks=dim,
                    block_shape=(1, 1),
                    cond_dim=cond_dim,
                ),
            )
        else:
            keys = random.split(key, depth + 1)

            block_shapes = [
                (block_dim, 1),
                *[(block_dim, block_dim)] * (depth - 1),
                (1, block_dim),
            ]
            cond_dims = [cond_dim] + [None] * depth

            for layer_key, block_shape, cond_d in zip(
                keys,
                block_shapes,
                cond_dims,
                strict=True,
            ):
                layers.append(
                    BlockAutoregressiveLinear(
                        layer_key,
                        n_blocks=dim,
                        block_shape=block_shape,
                        cond_dim=cond_d,
                    ),
                )

        self.depth = depth
        self.layers = layers
        self.block_dim = block_dim
        self.shape = (dim,)
        self.cond_shape = (cond_dim,) if cond_dim is not None else None
        self.activation = activation

    def transform(self, x, condition=None):
        for layer in self.layers[:-1]:
            x = layer(x, condition)[0]
            x = eqx.filter_vmap(self.activation.transform)(x)
            condition = None
        return self.layers[-1](x, condition)[0]

    def transform_and_log_det(self, x, condition=None):
        log_jacobian_3ds = []
        for layer in self.layers[:-1]:
            x, log_det_3d = layer(x, condition)
            log_jacobian_3ds.append(log_det_3d)

            x, log_det_3d = self._activation_and_log_det_3d(x)
            log_jacobian_3ds.append(log_det_3d)
            condition = None  # only pass array condition to first layer

        x, log_det_3d = self.layers[-1](x, condition)
        log_jacobian_3ds.append(log_det_3d)

        log_det = log_jacobian_3ds[-1]
        for log_jacobian in reversed(log_jacobian_3ds[:-1]):
            log_det = logmatmulexp(log_det, log_jacobian)
        return x, log_det.sum()

    def inverse(self, *args, **kwargs):
        raise NotImplementedError(
            "This transform would require numerical methods for inversion.",
        )

    def inverse_and_log_det(self, *args, **kwargs):
        raise NotImplementedError(
            "This transform would require numerical methods for inversion.",
        )

    def _activation_and_log_det_3d(self, x):
        """Compute activation and the log determinant (blocks, block_dim, block_dim)."""
        x, log_abs_grads = eqx.filter_vmap(self.activation.transform_and_log_det)(x)
        log_det_3d = jnp.full((self.shape[0], self.block_dim, self.block_dim), -jnp.inf)
        log_det_3d = log_det_3d.at[
            :,
            jnp.arange(self.block_dim),
            jnp.arange(self.block_dim),
        ].set(log_abs_grads.reshape(self.shape[0], self.block_dim))
        return x, log_det_3d


def logmatmulexp(x, y):
    """Numerically stable version of ``(x.log() @ y.log()).exp()``.

    From numpyro https://github.com/pyro-ppl/numpyro/blob/
    f2ff89a3a7147617e185eb51148eb15d56d44661/numpyro/distributions/util.py#L387.
    """
    x_shift = jax.lax.stop_gradient(jnp.amax(x, -1, keepdims=True))
    y_shift = jax.lax.stop_gradient(jnp.amax(y, -2, keepdims=True))
    xy = jnp.log(jnp.matmul(jnp.exp(x - x_shift), jnp.exp(y - y_shift)))
    return xy + x_shift + y_shift
