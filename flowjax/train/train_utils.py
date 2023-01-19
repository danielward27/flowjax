from typing import Sequence, List
from flowjax.utils import Array
import jax.random as jr
import jax.numpy as jnp


def train_val_split(key: jr.KeyArray, arrays: Sequence[Array], val_prop: float = 0.1):
    """Train validation split along axis 0.

    Args:
        key (KeyArray): Jax PRNGKey
        arrays List[Array]: List of arrays.
        val_prop (float, optional): Proportion of data to use for validation. Defaults to 0.1.

    Returns:
        Tuple[Tuple]: (train_arrays, validation_arrays)
    """
    if not (0 <= val_prop <= 1):
        raise ValueError("val_prop should be between 0 and 1.")
    n = arrays[0].shape[0]
    permutation = jr.permutation(key, jnp.arange(n))
    arrays = tuple(a[permutation] for a in arrays)
    n_train = n - round(val_prop * n)
    train = tuple(a[:n_train] for a in arrays)
    val = tuple(a[n_train:] for a in arrays)
    return train, val


def count_fruitless(losses: List[float]) -> int:
    """Given a list of losses from each epoch, count the number of epochs since
    the minimum loss.

    Args:
        losses (List[float]): List of losses.

    """
    min_idx = jnp.argmin(jnp.array(losses)).item()
    return len(losses) - min_idx - 1
