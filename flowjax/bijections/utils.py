"""Utility bijections (embedding network, permutations, inversion etc.)."""
from typing import Callable, ClassVar

import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.typing import ArrayLike

from flowjax.bijections.bijection import AbstractBijection


class Invert(AbstractBijection):
    """Invert a bijection.

    This wraps a bijection, such that the transform methods become the inverse
    methods and vice versa. Note that in general, we define bijections such that
    the forward methods are preffered, i.e. faster/actually implemented. For
    training flows, we generally want the inverse method (used in density
    evaluation), to be faster. Hence it is often useful to use this class to
    achieve this aim.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    bijection: AbstractBijection

    def __init__(self, bijection: AbstractBijection):
        """Initialize the bijection.

        Args:
            bijection (AbstractBijection): Bijection to invert.
        """
        self.bijection = bijection
        self.shape = bijection.shape
        self.cond_shape = bijection.cond_shape

    def transform(self, x, condition=None):
        return self.bijection.inverse(x, condition)

    def transform_and_log_det(self, x, condition=None):
        return self.bijection.inverse_and_log_det(x, condition)

    def inverse(self, y, condition=None):
        return self.bijection.transform(y, condition)

    def inverse_and_log_det(self, y, condition=None):
        return self.bijection.transform_and_log_det(y, condition)


class Permute(AbstractBijection):
    """Permutation transformation."""

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    permutation: tuple[Array, ...]
    inverse_permutation: tuple[Array, ...]

    def __init__(self, permutation: ArrayLike):
        """Initialize the permutation bijection.

        Args:
            permutation (ArrayLike): An array with shape matching the array to
                transform, with elements 0-(array.size-1) representing the new order
                based on the flattened array (uses, C-like ordering).
        """
        permutation = jnp.asarray(permutation)
        checkify.check(
            (permutation.ravel().sort() == jnp.arange(permutation.size)).all(),
            "Invalid permutation array provided.",
        )
        self.shape = permutation.shape

        indices = jnp.unravel_index(permutation.ravel(), permutation.shape)
        self.permutation = tuple(jnp.reshape(i, permutation.shape) for i in indices)

        inv_indices = jnp.unravel_index(
            jnp.argsort(permutation.ravel()), permutation.shape
        )
        self.inverse_permutation = tuple(
            jnp.reshape(i, permutation.shape) for i in inv_indices
        )

    def transform(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return x[self.permutation]

    def transform_and_log_det(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return x[self.permutation], jnp.array(0)

    def inverse(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        return y[self.inverse_permutation]

    def inverse_and_log_det(self, y, condition=None):
        x, _ = self._argcheck_and_cast(y)
        return y[self.inverse_permutation], jnp.array(0)


class Flip(AbstractBijection):
    """Flip the input array. Condition argument is ignored.

    Args:
        shape (tuple[int, ...]): The shape of the bijection.
            Defaults to None.
    """

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def transform(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return jnp.flip(x)

    def transform_and_log_det(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return jnp.flip(x), jnp.array(0)

    def inverse(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        return jnp.flip(y)

    def inverse_and_log_det(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        return jnp.flip(y), jnp.array(0)


class Partial(AbstractBijection):
    """Applies bijection to specific indices of an input."""

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    bijection: AbstractBijection
    idxs: int | slice | Array | tuple

    def __init__(
        self,
        bijection: AbstractBijection,
        idxs: int | slice | Array | tuple,
        shape: tuple[int, ...],
    ):
        """Initialize the bijection.

        Args:
            bijection (AbstractBijection): Bijection that is compatible with the subset
                of x indexed by idxs. idxs: Indices (Integer, a slice, or an ndarray
                with integer/bool dtype) of the transformed portion.
            idxs (int | slice | Array | tuple): The indexes to transform.
            shape (tuple[int, ...] | None): Shape of the bijection. Defaults to None.
        """
        self.bijection = bijection
        self.idxs = idxs
        self.shape = shape
        self.cond_shape = bijection.cond_shape

        if jnp.zeros(shape)[idxs].shape != bijection.shape:
            raise ValueError(
                f"The bijection shape is incompatible with the subset of the input "
                f"indexed by 'idxs'. The bijection has a shape of {bijection.shape}, "
                f"while the subset has a shape of {jnp.zeros(shape)[idxs].shape}."
            )

    def transform(self, x, condition=None):
        x, condition = self._argcheck_and_cast(x, condition)
        y = self.bijection.transform(x[self.idxs], condition)
        return x.at[self.idxs].set(y)

    def transform_and_log_det(self, x, condition=None):
        x, condition = self._argcheck_and_cast(x, condition)
        y, log_det = self.bijection.transform_and_log_det(x[self.idxs], condition)
        return x.at[self.idxs].set(y), log_det

    def inverse(self, y, condition=None):
        y, condition = self._argcheck_and_cast(y, condition)
        x = self.bijection.inverse(y[self.idxs], condition)
        return y.at[self.idxs].set(x)

    def inverse_and_log_det(self, y, condition=None):
        y, condition = self._argcheck_and_cast(y, condition)
        x, log_det = self.bijection.inverse_and_log_det(y[self.idxs], condition)
        return y.at[self.idxs].set(x), log_det


class EmbedCondition(AbstractBijection):
    """Wrap a bijection to include an embedding network.

    Generally this is used to reduce the dimensionality of the conditioning
    variable. The returned bijection has cond_dim equal to the raw condition size.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    bijection: AbstractBijection
    embedding_net: Callable

    def __init__(
        self,
        bijection: AbstractBijection,
        embedding_net: Callable,
        raw_cond_shape: tuple[int, ...],
    ) -> None:
        """Intialize the bijection.

        Args:
            bijection (AbstractBijection): Bijection with ``bijection.cond_dim`` equal
            to the embedded size.
            embedding_net (Callable): A callable (e.g. equinox module) that embeds a
            conditioning variable to size ``bijection.cond_dim``.
            raw_cond_shape (tuple[int, ...] | None): The dimension of the raw
            conditioning variable.
        """
        self.bijection = bijection
        self.embedding_net = embedding_net

        self.shape = bijection.shape
        self.cond_shape = raw_cond_shape

    def transform(self, x, condition=None):
        x, condition = self._argcheck_and_cast(x, condition)
        condition = self.embedding_net(condition)
        return self.bijection.transform(x, condition)

    def transform_and_log_det(self, x, condition=None):
        x, condition = self._argcheck_and_cast(x, condition)
        condition = self.embedding_net(condition)
        return self.bijection.transform_and_log_det(x, condition)

    def inverse(self, y, condition=None):
        y, condition = self._argcheck_and_cast(y, condition)
        condition = self.embedding_net(condition)
        return self.bijection.inverse(y, condition)

    def inverse_and_log_det(self, y, condition=None):
        y, condition = self._argcheck_and_cast(y, condition)
        condition = self.embedding_net(condition)
        return self.bijection.inverse_and_log_det(y, condition)
