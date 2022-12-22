from typing import Any
import jax
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


def real_to_increasing_on_interval(
    arr: Array, B: float = 1, softmax_adjust: float = 1e-2
):
    """Transform unconstrained parameter vector to monotonically increasing positions on [-B, B].

    Args:
        arr (Array): Parameter vector.
        B (float, optional): Interval to transform output. Defaults to 1.
        softmax_adjust (float, optional): Rescales softmax output using (widths + softmax_adjust/widths.size) / (1 + softmax_adjust). e.g. 0=no adjustment, 1=average softmax output with evenly spaced widths, >1 promotes more evenly spaced widths.
    """
    if softmax_adjust < 0:
        raise ValueError("softmax_adjust should be >= 0.")
    widths = jax.nn.softmax(arr)
    widths = (widths + softmax_adjust / widths.size) / (1 + softmax_adjust)
    widths = widths.at[0].set(widths[0] / 2)
    return 2 * B * jnp.cumsum(widths) - B
