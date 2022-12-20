from typing import Any

import jax.numpy as jnp

Array = Any  # TODO Can use TypeAlias from typing in future.

def _identity(x):
    return x


def tile_until_length(x: Array, max_len: int):
    """Tile array along until a maximum length is reached. If x.ndim > 1,
    then the array is flattened, and a flattened array is returned.

    Args:
        x (Array): Input array.
        max_len (int): Maximum length of output array.
    """
    x = jnp.ravel(x)
    num_reps = max_len // len(x) + 1
    y = jnp.tile(x, num_reps)
    return y[:max_len]


def broadcast_arrays_1d(*args):
    "Broadcast arrays, with all outputs being 1d."
    args = jnp.broadcast_arrays(*args)
    return [promote_to_1d(a) for a in args]


def promote_to_1d(arr: Array):
    if jnp.ndim(arr) > 1:
        raise ValueError(f"Cannot convert array with shape {arr.shape} to 1d.")
    else:
        return jnp.atleast_1d(arr)
