from typing import Any, Sequence
import jax
import jax.numpy as jnp
import equinox as eqx
import jax.flatten_util as jfu

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
    return x - jnp.pad(x[:-1], (1, 0))


def merge_shapes(shapes: Sequence):
    """ "Broadcast shapes used in bijections and distributions.

    Namely:
        - A shape None is used to mean any shape (either unknown, unimportant, or compatible with any shape).
        - We require all shapes that are not None to have the same length, and matching values.
    """  # TODO update these docs.
    if len(shapes) == 0:
        raise ValueError("No shapes have been provided.")
    elif all(s is None for s in shapes):
        return None
    else:
        shapes = [s for s in shapes if s is not None]
        if all(s == shapes[0] for s in shapes):
            return shapes[0]
        else:
            raise ValueError("The shapes do not match.")


def check_shapes(shapes):  # TODO test this?
    shapes = [s for s in shapes if s is not None]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError("The shapes do not match.")


def _get_ufunc_signature(in_shapes, out_shapes):
    """Convert a sequence of in_shapes, and out_shapes to a universal function signature.
    
    Example:
        >>> _get_ufunc_signature([(3,),(2,3)], [()])
        "(3),(2,3)->()"
    """
    def _shapes_to_str(shapes):
        result = [str(s) if len(s)!=1 else str(s).replace(",", "") for s in shapes]
        return ",".join(result).replace(" ", "")

    in_shapes_str = _shapes_to_str(in_shapes)
    out_shapes_str = _shapes_to_str(out_shapes)
    return f"{in_shapes_str}->{out_shapes_str}"




def get_ravelled_bijection_constructor(bijection, filter_spec=eqx.is_inexact_array):
    """Given a bijection, returns a tuple containing
    1) a constructor for the bijection from a flattened array; 2) the current flattened
    parameters.

    Args:
        bijection (Bijection): Bijection.
        filter_spec: Filter function. Defaults to eqx.is_inexact_array.
    """

    params, static = eqx.partition(bijection, filter_spec)
    bias, unravel = jfu.ravel_pytree(params)

    def f(ravelled_params: Array):
        ravelled_params = ravelled_params + bias
        params = unravel(ravelled_params)
        return eqx.combine(params, static)

    return f, bias
