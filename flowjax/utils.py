from typing import Any, Sequence
import jax
import jax.numpy as jnp
import equinox as eqx
import jax.flatten_util as jfu
# from flowjax.bijections import Bijection

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

def inv_cum_sum(x):
    return x - jnp.pad(x[:-1], (1,0))

def increasing_on_interval_to_real(
    arr: Array, B: float = 1, softmax_adjust: float = 1e-2
):
    widths = inv_cum_sum(arr/(2*B)) + B
    widths = widths.at[0].set(widths[0]*2)
    widths = widths*(1 + softmax_adjust) - softmax_adjust / widths.size
    # TODO finish

def replace_bijection_params_from_flat(bijection, flat, filter_spec = eqx.is_inexact_array):
    old_params, static = eqx.partition(bijection, filter_spec)
    _, unravel = jfu.ravel_pytree(old_params)
    new_params = unravel(flat)
    return eqx.combine(new_params, static)



def broadcast_shapes(shapes: Sequence):  # TODO rename to avoid name clash with jnp.
    """"Broadcast shapes used in bijections and distributions. Note we use different rules to numpy broadcasting.

    Namely:
        - A shape None is used to mean any shape (either unknown, unimportant, or compatible with any shape).
        - An element in a shape of -1 acts as a "wildcard", indicating the bijection can act on any length on the corresponding array dimension.
        - We require all shapes that are not None to have the same length, and matching values (with the exception of the wildcard).
    """  # TODO update these docs.
    if len(shapes) == 0:
        raise ValueError("No shapes have been provided.")
    elif all(s is None for s in shapes):
        return None
    else:
        shapes = [s for s in shapes if s is not None]
        if all(s==shapes[0] for s in shapes):
            return shapes[0]
        else:
            raise ValueError("The shapes do not match.")

def check_shapes(shapes):  # TODO test this?
    shapes = [s for s in shapes if s is not None]
    if not all(s==shapes[0] for s in shapes):
        raise ValueError("The shapes do not match.")


def _get_ufunc_signature(in_shapes, out_shapes):

    def _shapes_to_str(shapes):
        result = [] 
        for shape in shapes:
            shape_str = f"({','.join([str(s) for s in shape])})"
            result.append(shape_str)
        return ",".join(result)

    in_shapes_str = _shapes_to_str(in_shapes)
    out_shapes_str = _shapes_to_str(out_shapes)
    return f"{in_shapes_str}->{out_shapes_str}"
