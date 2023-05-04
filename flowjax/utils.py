"""Utility functions."""
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from flowjax.bijections.bijection import Bijection


def real_to_increasing_on_interval(
    arr: Array, B: float = 1, softmax_adjust: float = 1e-2
):
    """Transform unconstrained parameter vector to monotonically increasing positions on [-B, B].

    Args:
        arr (Array): Parameter vector.
        B (float): Interval to transform output. Defaults to 1.
        softmax_adjust (float): Rescales softmax output using (widths +
            softmax_adjust/widths.size) / (1 + softmax_adjust). e.g. 0=no adjustment,
            1=average softmax output with evenly spaced widths, >1 promotes more evenly
            spaced widths.
    """
    if softmax_adjust < 0:
        raise ValueError("softmax_adjust should be >= 0.")
    widths = jax.nn.softmax(arr)  # type: ignore
    widths = (widths + softmax_adjust / widths.size) / (1 + softmax_adjust)
    widths = widths.at[0].set(widths[0] / 2)
    return 2 * B * jnp.cumsum(widths) - B


def inv_cum_sum(x):
    """Inverse of cumulative sum operation."""
    return x - jnp.pad(x[:-1], (1, 0))


def merge_cond_shapes(shapes: Sequence):
    """Merges shapes (tuples of ints or None) used in bijections and distributions.
    Returns None if all shapes are None, otherwise checks the shapes match, and returns
    the shape.
    """
    if len(shapes) == 0:
        raise ValueError("No shapes have been provided.")
    if all(s is None for s in shapes):
        return None
    shapes = [s for s in shapes if s is not None]
    if all(s == shapes[0] for s in shapes):
        return shapes[0]
    raise ValueError("The shapes do not match.")


def check_shapes_match(shapes: list[tuple[int, ...]]):
    """Check shapes match and produce a useful error message."""

    for i, shape in enumerate(shapes):
        if shape != shapes[0]:
            raise ValueError(
                f"Expected shapes to match, but index 0 had shape {shapes[0]}, and "
                f"index {i} had shape {shape}."
            )


def _get_ufunc_signature(in_shapes, out_shapes):
    """Convert a sequence of in_shapes, and out_shapes to a universal function signature.

    Example:
        >>> _get_ufunc_signature([(3,),(2,3)], [()])
        "(3),(2,3)->()"
    """

    def _shapes_to_str(shapes):
        result = [str(s) if len(s) != 1 else str(s).replace(",", "") for s in shapes]
        return ",".join(result).replace(" ", "")

    in_shapes_str = _shapes_to_str(in_shapes)
    out_shapes_str = _shapes_to_str(out_shapes)
    return f"{in_shapes_str}->{out_shapes_str}"


def get_ravelled_bijection_constructor(
    bijection: Bijection, filter_spec=eqx.is_inexact_array
):
    """Given a bijection, returns a tuple containing
    1) a constructor for the bijection from a flattened array; 2) the current flattened
    parameters.

    Args:
        bijection (Bijection): Bijection.
        filter_spec: Filter function. Defaults to eqx.is_inexact_array.
    """
    params, static = eqx.partition(bijection, filter_spec)  # type: ignore
    current, unravel = ravel_pytree(params)

    def constructor(ravelled_params: Array):
        params = unravel(ravelled_params)
        return eqx.combine(params, static)

    return constructor, current
