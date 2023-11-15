"""Utility bijections (embedding network, permutations, inversion etc.)."""
from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.typing import ArrayLike

from flowjax.bijections.bijection import AbstractBijection
from flowjax.utils import arraylike_to_array


class Invert(AbstractBijection):
    """Invert a bijection.

    This wraps a bijection, such that the transform methods become the inverse
    methods and vice versa. Note that in general, we define bijections such that
    the forward methods are preffered, i.e. faster/actually implemented. For
    training flows, we generally want the inverse method (used in density
    evaluation), to be faster. Hence it is often useful to use this class to
    achieve this aim.

    Args:
        bijection: Bijection to invert.
    """

    bijection: AbstractBijection

    def transform(self, x, condition=None):
        return self.bijection.inverse(x, condition)

    def transform_and_log_det(self, x, condition=None):
        return self.bijection.inverse_and_log_det(x, condition)

    def inverse(self, y, condition=None):
        return self.bijection.transform(y, condition)

    def inverse_and_log_det(self, y, condition=None):
        return self.bijection.transform_and_log_det(y, condition)

    @property
    def shape(self):
        return self.bijection.shape

    @property
    def cond_shape(self):
        return self.bijection.cond_shape


class Permute(AbstractBijection):
    """Permutation transformation.

    Args:
        permutation: An array with shape matching the array to transform, with elements
            0-(array.size-1) representing the new order based on the flattened array
            (uses, C-like ordering).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    permutation: tuple[Array, ...]
    inverse_permutation: tuple[Array, ...]

    def __init__(self, permutation: ArrayLike):
        permutation = arraylike_to_array(permutation)
        checkify.check(
            (permutation.ravel().sort() == jnp.arange(permutation.size)).all(),
            "Invalid permutation array provided.",
        )
        self.shape = permutation.shape

        indices = jnp.unravel_index(permutation.ravel(), permutation.shape)
        self.permutation = tuple(jnp.reshape(i, permutation.shape) for i in indices)

        inv_indices = jnp.unravel_index(
            jnp.argsort(permutation.ravel()),
            permutation.shape,
        )
        self.inverse_permutation = tuple(
            jnp.reshape(i, permutation.shape) for i in inv_indices
        )

    def transform(self, x, condition=None):
        return x[self.permutation]

    def transform_and_log_det(self, x, condition=None):
        return x[self.permutation], jnp.array(0)

    def inverse(self, y, condition=None):
        return y[self.inverse_permutation]

    def inverse_and_log_det(self, y, condition=None):
        return y[self.inverse_permutation], jnp.array(0)


class Flip(AbstractBijection):
    """Flip the input array. Condition argument is ignored.

    Args:
        shape: The shape of the bijection. Defaults to None.
    """

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def transform(self, x, condition=None):
        return jnp.flip(x)

    def transform_and_log_det(self, x, condition=None):
        return jnp.flip(x), jnp.array(0)

    def inverse(self, y, condition=None):
        return jnp.flip(y)

    def inverse_and_log_det(self, y, condition=None):
        return jnp.flip(y), jnp.array(0)


class Partial(AbstractBijection):
    """Applies bijection to specific indices of an input.

    Args:
        bijection: Bijection that is compatible with the subset
            of x indexed by idxs. idxs: Indices (Integer, a slice, or an ndarray
            with integer/bool dtype) of the transformed portion.
        idxs: The indexes to transform.
        shape: Shape of the bijection. Defaults to None.
    """

    bijection: AbstractBijection
    idxs: int | slice | Array | tuple
    shape: tuple[int, ...]

    def __check_init__(self):
        expected_shape = jnp.zeros(self.shape)[self.idxs].shape
        if expected_shape != self.bijection.shape:
            raise ValueError(
                f"The bijection shape is incompatible with the subset of the input "
                f"indexed by 'idxs'. Bijection has shape {self.bijection.shape}, "
                f"while the subset has a shape of {expected_shape}.",
            )

    def transform(self, x, condition=None):
        y = self.bijection.transform(x[self.idxs], condition)
        return x.at[self.idxs].set(y)

    def transform_and_log_det(self, x, condition=None):
        y, log_det = self.bijection.transform_and_log_det(x[self.idxs], condition)
        return x.at[self.idxs].set(y), log_det

    def inverse(self, y, condition=None):
        x = self.bijection.inverse(y[self.idxs], condition)
        return y.at[self.idxs].set(x)

    def inverse_and_log_det(self, y, condition=None):
        x, log_det = self.bijection.inverse_and_log_det(y[self.idxs], condition)
        return y.at[self.idxs].set(x), log_det

    @property
    def cond_shape(self):
        return self.bijection.cond_shape


class Identity(AbstractBijection):
    """The identity bijection.

    Args:
       shape: The shape of the bijection. Defaults to ().
    """

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def transform(self, x, condition=None):
        return x

    def transform_and_log_det(self, x, condition=None):
        return x, jnp.zeros(())

    def inverse(self, y, condition=None):
        return y

    def inverse_and_log_det(self, y, condition=None):
        return y, jnp.zeros(())


class EmbedCondition(AbstractBijection):
    """Wrap a bijection to include an embedding network.

    Generally this is used to reduce the dimensionality of the conditioning
    variable. The returned bijection has cond_dim equal to the raw condition size.

    Args:
        bijection: Bijection with ``bijection.cond_dim`` equal to the embedded size.
        embedding_net: A callable (e.g. equinox module) that embeds a conditioning
            variable to size ``bijection.cond_dim``.
        raw_cond_shape: The dimension of the raw conditioning variable.
    """

    bijection: AbstractBijection
    cond_shape: tuple[int, ...]
    embedding_net: Callable

    def __init__(
        self,
        bijection: AbstractBijection,
        embedding_net: Callable,
        raw_cond_shape: tuple[int, ...],
    ) -> None:
        self.bijection = bijection
        self.embedding_net = embedding_net
        self.cond_shape = raw_cond_shape

    def transform(self, x, condition=None):
        condition = self.embedding_net(condition)
        return self.bijection.transform(x, condition)

    def transform_and_log_det(self, x, condition=None):
        condition = self.embedding_net(condition)
        return self.bijection.transform_and_log_det(x, condition)

    def inverse(self, y, condition=None):
        condition = self.embedding_net(condition)
        return self.bijection.inverse(y, condition)

    def inverse_and_log_det(self, y, condition=None):
        condition = self.embedding_net(condition)
        return self.bijection.inverse_and_log_det(y, condition)

    @property
    def shape(self):
        return self.bijection.shape
