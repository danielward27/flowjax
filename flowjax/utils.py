"""Utility functions."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike

import flowjax


class _VectorizedBijection:
    """Wrap a flowjax bijection to support vectorization.

    Args:
        bijection: flowjax bijection to be wrapped.
    """

    def __init__(self, bijection: flowjax.bijections.AbstractBijection):
        self.bijection = bijection
        self.shape = self.bijection.shape
        self.cond_shape = self.bijection.cond_shape

    def transform(self, x, condition=None):
        transform = self.vectorize(self.bijection.transform)
        return transform(x, condition)

    def inverse(self, y, condition=None):
        inverse = self.vectorize(self.bijection.inverse)
        return inverse(y, condition)

    def transform_and_log_det(self, x, condition=None):
        transform_and_log_det = self.vectorize(
            self.bijection.transform_and_log_det,
            log_det=True,
        )
        return transform_and_log_det(x, condition)

    def inverse_and_log_det(self, x, condition=None):
        inverse_and_log_det = self.vectorize(
            self.bijection.inverse_and_log_det,
            log_det=True,
        )
        return inverse_and_log_det(x, condition)

    def vectorize(self, func, *, log_det=False):
        in_shapes, out_shapes = [self.bijection.shape], [self.bijection.shape]
        if log_det:
            out_shapes.append(())
        if self.bijection.cond_shape is not None:
            in_shapes.append(self.bijection.cond_shape)
            exclude = frozenset()
        else:
            exclude = frozenset([1])
        sig = _get_ufunc_signature(in_shapes, out_shapes)
        return jnp.vectorize(func, signature=sig, excluded=exclude)


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


def get_ravelled_pytree_constructor(tree, filter_spec=eqx.is_inexact_array) -> tuple:
    """Get a pytree constructor taking ravelled parameters, and the number of params.

    The constructor takes a single argument as input, which is all the bijection
    parameters flattened into a single contiguous vector. This is useful when we wish to
    parameterize a pytree with a neural neural network. Calling the constructor
    at the zero vector will return the initial pytree. Parameters warpped in
    ``NonTrainable`` are treated as leaves during partitioning.

    Args:
        tree: Pytree to form constructor for.
        filter_spec: Filter function to specify parameters. Defaults to
            eqx.is_inexact_array.

    Returns:
        tuple: Tuple containing the constructor, and the number of parameters.
    """
    params, static = eqx.partition(
        tree,
        filter_spec,
        is_leaf=lambda leaf: isinstance(leaf, flowjax.wrappers.NonTrainable),
    )
    init, unravel = ravel_pytree(params)

    def constructor(ravelled_params: Array):
        params = unravel(ravelled_params + init)
        return eqx.combine(params, static)

    return constructor, len(init)


def arraylike_to_array(arr: ArrayLike, err_name: str = "input", **kwargs) -> Array:
    """Check the input is arraylike and convert to a jax Array with ``jnp.asarray``.

    Combines ``jnp.asarray``, with an isinstance(arr, ArrayLike) check. This
    allows inputs to be jax arrays, numpy arrays, python built in numeric types
    (float, int) etc, but does not allow list or tuple inputs (which are not arraylike
    and can introduce overhead and confusing behaviour in certain cases).

    Args:
        arr: Arraylike input to convert to a jax array.
        err_name: Name of the input in the error message. Defaults to "input".
        **kwargs: Keyword arguments passed to jnp.asarray.
    """
    if not isinstance(arr, ArrayLike):
        raise TypeError(
            f"Expected {err_name} to be arraylike; got {type(arr).__name__}.",
        )
    return jnp.asarray(arr, **kwargs)
