import jax.numpy as jnp
from typing import Any

Array = Any  # Custom type for Arrays (clearer typehint)


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
