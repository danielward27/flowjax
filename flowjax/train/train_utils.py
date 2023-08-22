"""Utility functions for training."""
from functools import partial
from typing import Sequence

import jax.numpy as jnp
import jax.random as jr
from jax import Array, jit


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


def get_batches(
    arrays: Sequence[Array], batch_size: int, axis: int | Sequence[int] = 0
):
    """Reshape a sequence of arrays to have a leading dimension of size
    ``data_len // batch_size``, i.e. the number of batches. Batching occurs along the axis
    specified, e.g. for axis=0, each input array would be reshaped to
    ``(num_batches, batch_size, *array.shape[1:])``.

    Considerations:
        - The values in the last batch are dropped if truncated, i.e. if
            ``data_len % batch_size != 0``.
        - If the batch size is greater than the length of the data, then we set the batch
            size to be equal to the data length.

    Args:
        arrays (Sequence[Array]): Sequence of arrays, with shape matching on specified
            axis.
        batch_size (int): The batch size.
        axis (int | Sequence[int], optional): Axis or a list of axes, denoting the axis
            along which to batch. Defaults to 0.
    """
    if isinstance(axis, int):
        axis = [axis] * len(arrays)

    data_len = arrays[0].shape[axis[0]]
    if not all(arr.shape[ax] == data_len for arr, ax in zip(arrays, axis)):
        raise ValueError("Array dimensions do not match along the batch axis.")

    return tuple(_add_batch(arr, batch_size, ax) for arr, ax in zip(arrays, axis))


@partial(jit, static_argnums=[1, 2])
def _add_batch(array, batch_size, axis: int = 0):
    "Adds a leading dimension for batches."
    data_len = array.shape[axis]
    batch_size = min(batch_size, data_len)
    num_batches = data_len // batch_size
    array = jnp.moveaxis(array, axis, 0)  # Batch axis to 0
    data_shape = array.shape[1:]
    array = array[: num_batches * batch_size]
    array = array.reshape(num_batches, batch_size, *data_shape)
    array = jnp.moveaxis(array, 1, axis + 1)  # Restore batch axis to original position
    return array
