"""Utility functions for training."""

from collections.abc import Callable, Sequence
from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import jit
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar, Shaped


@eqx.filter_jit
def step(
    params: PyTree,
    *args,
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    loss_fn: Callable[[PyTree, PyTree], Scalar],
    **kwargs,
):
    """Carry out a training step.

    Args:
        params: Parameters for the model
        *args: Arguments passed to the loss function (often the static components
            of the model).
        optimizer: Optax optimizer.
        opt_state: Optimizer state.
        loss_fn: The loss function. This should take params and static as the first two
            arguments.
        **kwargs: Key word arguments passed to the loss function.

    Returns:
        tuple: (params, opt_state, loss_val)
    """
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
        params,
        *args,
        **kwargs,
    )
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_val


def train_val_split(
    key: PRNGKeyArray,
    arrays: Sequence[Shaped[Array, "batch ..."]],
    val_prop: float = 0.1,
):
    """Random train validation split for a sequence of arrays.

    Args:
        key: Jax random key.
        arrays: Sequence of arrays, with matching size on axis 0.
        val_prop: Proportion of data to use for validation. Defaults to 0.1.

    Returns:
        A tuple containing the train arrays and the validation arrays.
    """
    if not 0 <= val_prop <= 1:
        raise ValueError("val_prop should be between 0 and 1.")

    num_samples = arrays[0].shape[0]
    if not all(isinstance(a, Shaped[Array, " dim ..."]) for a in arrays):
        raise ValueError("Array dimensions must match along axis 0.")

    n_train = num_samples - round(val_prop * num_samples)
    arrays = [jr.permutation(key, a) for a in arrays]
    train_arrays = [arr[:n_train] for arr in arrays]
    val_arrays = [arr[n_train:] for arr in arrays]
    return train_arrays, val_arrays


@partial(jit, static_argnums=1)
def get_batches(arrays: Sequence[Array], batch_size: int):
    """Reshape a sequence of arrays to have an additional dimension for the batches.

    Specifically, this will transform an array with shape ``(data_len, *rest)`` to
    ``(num_batches, batch_size, *rest)``.

    Considerations:
        - The values in the last batch are dropped if truncated, i.e. if
            ``data_len % batch_size != 0``.
        - If the batch size is greater than the length of the data, then we set the
            batch size to be equal to the data length.

    Args:
        arrays: Sequence of arrays, with shape matching on axis 0.
        batch_size: The batch size.
    """
    data_len = arrays[0].shape[0]
    if not all(arr.shape[0] == data_len for arr in arrays):
        raise ValueError("Array dimensions do not match along the batch axis.")

    return tuple(_add_batch(arr, batch_size) for arr in arrays)


def _add_batch(arr, batch_size):
    """Adds a leading dimension for batches, dropping the last batch if truncated."""
    batch_size = min(batch_size, arr.shape[0])
    n_batches = arr.shape[0] // batch_size
    return arr[: n_batches * batch_size].reshape(n_batches, batch_size, *arr.shape[1:])


def count_fruitless(losses: list[float]) -> int:
    """Count the number of epochs since the minimum loss in a list of losses.

    Args:
        losses: List of losses.

    """
    min_idx = jnp.argmin(jnp.array(losses)).item()
    return len(losses) - min_idx - 1
