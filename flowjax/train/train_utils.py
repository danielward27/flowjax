"""Utility functions for training."""
from typing import Sequence

import jax.numpy as jnp
import jax.random as jr
from jax import Array


def train_val_split(
    key: jr.KeyArray,
    arrays: Sequence[Array],
    val_prop: float = 0.1,
    axis: int | Sequence[int] = 0,
):
    """Random train validation split along a given axis.

    Args:
        key (KeyArray): Jax PRNGKey
        arrays Sequence[Array]: Sequence of arrays, with matching size on specified axis.
        val_prop (float): Proportion of data to use for validation. Defaults to 0.1.
        axis (int | Sequence[int]): Axis along which to carry out split. A sequence of
            integers matching the length of arrays can be provided to specify different
            axes for each array. Defaults to 0.

    Returns:
        tuple[tuple]: (train_arrays, validation_arrays)
    """
    if isinstance(axis, int):
        axis = [axis] * len(arrays)

    if not 0 <= val_prop <= 1:
        raise ValueError("val_prop should be between 0 and 1.")

    num_samples = arrays[0].shape[axis[0]]
    if not all(arr.shape[ax] == num_samples for arr, ax in zip(arrays, axis)):
        raise ValueError("Array dimensions do not match along specified axes.")

    permutation = jr.permutation(key, num_samples)
    n_train = num_samples - round(val_prop * num_samples)
    train_indices, val_indices = permutation[:n_train], permutation[n_train:]
    train_arrays = tuple(
        jnp.take(arr, train_indices, axis=ax) for arr, ax in zip(arrays, axis)
    )
    val_arrays = tuple(
        jnp.take(arr, val_indices, axis=ax) for arr, ax in zip(arrays, axis)
    )
    return train_arrays, val_arrays


def count_fruitless(losses: list[float]) -> int:
    """Given a list of losses from each epoch, count the number of epochs since
    the minimum loss.

    Args:
        losses (list[float]): List of losses.

    """
    min_idx = jnp.argmin(jnp.array(losses)).item()
    return len(losses) - min_idx - 1
