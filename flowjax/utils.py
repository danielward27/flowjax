"""Utility functions."""
from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike


def real_to_increasing_on_interval(
    arr: Array,
    B: float = 1,
    softmax_adjust: float = 1e-2,
):
    """Transform unconstrained vector to monotonically increasing positions on [-B, B].

    Args:
        arr: Parameter vector.
        B : Interval to transform output. Defaults to 1.
        softmax_adjust : Rescales softmax output using
            ``(widths + softmax_adjust/widths.size) / (1 + softmax_adjust)``. e.g.
            0=no adjustment, 1=average softmax output with evenly spaced widths, >1
            promotes more evenly spaced widths.
    """
    if softmax_adjust < 0:
        raise ValueError("softmax_adjust should be >= 0.")
    widths = jax.nn.softmax(arr)
    widths = (widths + softmax_adjust / widths.size) / (1 + softmax_adjust)
    widths = widths.at[0].set(widths[0] / 2)
    return 2 * B * jnp.cumsum(widths) - B


def inv_cum_sum(x):
    """Inverse of cumulative sum operation."""
    return x - jnp.pad(x[:-1], (1, 0))


def merge_cond_shapes(shapes: Sequence[tuple[int, ...] | None]):
    """Merges shapes (tuples of ints or None) used in bijections and distributions.

    Returns None if all shapes are None, otherwise checks none None shapes match, and
    returns the shape.
    """
    if len(shapes) == 0:
        raise ValueError("No shapes have been provided.")
    if all(s is None for s in shapes):
        return None
    shapes = [s for s in shapes if s is not None]
    if all(s == shapes[0] for s in shapes):
        return shapes[0]
    raise ValueError("The shapes do not match.")


def check_shapes_match(shapes: Sequence[tuple[int, ...]]):
    """Check shapes match and produce a useful error message."""
    for i, shape in enumerate(shapes):
        if shape != shapes[0]:
            raise ValueError(
                f"Expected shapes to match, but index 0 had shape {shapes[0]}, and "
                f"index {i} had shape {shape}.",
            )


def _get_ufunc_signature(in_shapes: tuple[int], out_shapes: tuple[int]):
    """Convert a sequence of in_shapes and out_shapes to a universal function signature.

    Example:
        >>> _get_ufunc_signature([(3,),(2,3)], [()])
        "(3),(2,3)->()"
    """

    def _shapes_to_str(shapes):
        result = (str(s) if len(s) != 1 else str(s).replace(",", "") for s in shapes)
        return ",".join(result).replace(" ", "")

    in_shapes_str = _shapes_to_str(in_shapes)
    out_shapes_str = _shapes_to_str(out_shapes)
    return f"{in_shapes_str}->{out_shapes_str}"


def get_ravelled_bijection_constructor(
    bijection,
    filter_spec=eqx.is_inexact_array,
) -> tuple:
    """Get a constructor taking ravelled parameters and the current ravelled parameters.

    The constructor takes a single argument as input, which is all the bijection
    parameters flattened into a single contiguous vector. This is useful when we wish to
    parameterize a bijection with a neural neural network, as it allows convenient
    construction of the bijection directly from the neural network output.

    Args:
        bijection: Bijection to form constructor for.
        filter_spec: Filter function to specify parameters. Defaults to
            eqx.is_inexact_array.

    Returns:
        The constructor, and the current parameter vector.
    """
    params, static = eqx.partition(bijection, filter_spec)
    current, unravel = ravel_pytree(params)

    def constructor(ravelled_params: Array):
        params = unravel(ravelled_params)
        return eqx.combine(params, static)

    return constructor, current


def arraylike_to_array(arr: ArrayLike, err_name: str = "input", **kwargs) -> Array:
    """Check the input is arraylike and convert to a jax Array with ``jnp.asarray``.

    Combines ``jnp.asarray``, with an isinstance(arr, ArrayLike) check. This
    allows inputs to be jax arrays, numpy arrays, python built in numeric types
    (float, int) etc, but does not allow list or tuple inputs (which are not arraylike
    and can introduce overhead and confusing behaviour in certain cases).

    Args:
        arr: Arraylike input to convert to a jax array.
        err_name: Name of the input in the error message. Defaults to "input".
        **kwargs: Key word arguments passed to jnp.asarray.
    """
    if not isinstance(arr, ArrayLike):
        raise TypeError(
            f"Expected {err_name} to be arraylike; got {type(arr).__name__}.",
        )
    return jnp.asarray(arr, **kwargs)
