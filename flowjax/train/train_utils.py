"""Utility functions for training."""
from typing import Sequence

import jax.numpy as jnp
import jax.random as jr
from jax import Array


def train_val_split(
    key: jr.KeyArray, arrays: Sequence[Array], val_prop: float = 0.1, axis: int = 0
):
    """Random train validation split along a given axis.

    Args:
        key (KeyArray): Jax PRNGKey
        arrays Sequence[Array]: Sequence of arrays, with matching size on specified axis.
        val_prop (float): Proportion of data to use for validation. Defaults to 0.1.
        axis (int): Axis along which to split. Defaults to 0.

    Returns:
        tuple[tuple]: (train_arrays, validation_arrays)
    """
    if not 0 <= val_prop <= 1:
        raise ValueError("val_prop should be between 0 and 1.")
    num_samples = arrays[0].shape[axis]
    if not all(a[0].shape[axis] == num_samples for a in arrays):
        raise ValueError(f"Array dimensions do not match along axis {axis}.")
    permutation = jr.permutation(key, num_samples)
    n_train = num_samples - round(val_prop * num_samples)
    train_indices, val_indices = permutation[:n_train], permutation[n_train:]
    train_arrays = tuple(jnp.take(array, train_indices, axis=axis) for array in arrays)
    val_arrays = tuple(jnp.take(array, val_indices, axis=axis) for array in arrays)
    return train_arrays, val_arrays


def count_fruitless(losses: list[float]) -> int:
    """Given a list of losses from each epoch, count the number of epochs since
    the minimum loss.

    Args:
        losses (list[float]): List of losses.

    """
    min_idx = jnp.argmin(jnp.array(losses)).item()
    return len(losses) - min_idx - 1
