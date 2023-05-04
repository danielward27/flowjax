"""Utility functions for training."""
from typing import Sequence

import jax.numpy as jnp
import jax.random as jr
from jax import Array


def train_val_split(key: jr.KeyArray, arrays: Sequence[Array], val_prop: float = 0.1):
    """Train validation split along axis 0.

    Args:
        key (KeyArray): Jax PRNGKey
        arrays Sequence[Array]: Sequence of arrays.
        val_prop (float): Proportion of data to use for validation. Defaults to 0.1.

    Returns:
        tuple[tuple]: (train_arrays, validation_arrays)
    """
    if not 0 <= val_prop <= 1:
        raise ValueError("val_prop should be between 0 and 1.")
    size_axis_0 = arrays[0].shape[0]
    permutation = jr.permutation(key, jnp.arange(size_axis_0))
    arrays = tuple(a[permutation] for a in arrays)
    n_train = size_axis_0 - round(val_prop * size_axis_0)
    train = tuple(a[:n_train] for a in arrays)
    val = tuple(a[n_train:] for a in arrays)
    return train, val


def count_fruitless(losses: list[float]) -> int:
    """Given a list of losses from each epoch, count the number of epochs since
    the minimum loss.

    Args:
        losses (list[float]): List of losses.

    """
    min_idx = jnp.argmin(jnp.array(losses)).item()
    return len(losses) - min_idx - 1
