"""Utility functions."""

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves, tree_map
from jaxtyping import Array, ArrayLike, PyTree


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


def _get_ufunc_signature(
    in_shapes: Sequence[tuple[int, ...]],
    out_shapes: Sequence[tuple[int, ...]],
):
    """Convert a sequence of in_shapes and out_shapes to a universal function signature.

    Example:
        .. doctest

            >>> _get_ufunc_signature([(3,),(2,3)], [()])
            '(3),(2,3)->()'
    """

    def _shapes_to_str(shapes):
        result = (str(s) if len(s) != 1 else str(s).replace(",", "") for s in shapes)
        return ",".join(result).replace(" ", "")

    in_shapes_str = _shapes_to_str(in_shapes)
    out_shapes_str = _shapes_to_str(out_shapes)
    return f"{in_shapes_str}->{out_shapes_str}"


def get_ravelled_pytree_constructor(
    tree,
    *args,
    **kwargs,
) -> tuple:
    """Get a pytree constructor taking ravelled parameters, and the number of params.

    The constructor takes a single argument as input, which is all the bijection
    parameters flattened into a single contiguous vector. This is useful when we wish to
    parameterize a pytree with a neural neural network. Calling the constructor
    at the zero vector will return the initial pytree. When using, you may wish to
    specify ``NonTrainable`` nodes as leaves, using the ``is_leaf`` argument.

    Args:
        tree: Pytree to form constructor for.
        *args: Arguments passed to ``eqx.partition``.
        **kwargs: Key word arguments passed to ``eqx.partition``.

    Returns:
        tuple: Tuple containing the constructor, and the number of parameters.
    """
    params, static = eqx.partition(tree, *args, **kwargs)
    init, unravel = ravel_pytree(params)

    def constructor(ravelled_params: Array):
        params = unravel(ravelled_params + init)
        return eqx.combine(params, static)

    return constructor, len(init)


def arraylike_to_array(
    arr: ArrayLike | None, err_name: str = "input", **kwargs
) -> Array:
    """Check the input is arraylike and convert to a JAX Array with ``jnp.asarray``.

    Combines ``jnp.asarray``, with an isinstance(arr, ArrayLike) check. This
    allows inputs to be JAX arrays, numpy arrays, python built in numeric types
    (float, int) etc, but does not allow list or tuple inputs (which are not arraylike
    and can introduce overhead and confusing behaviour in certain cases).

    Args:
        arr: Arraylike input to convert to a JAX array.
        err_name: Name of the input in the error message. Defaults to "input".
        **kwargs: Keyword arguments passed to jnp.asarray.
    """
    if not isinstance(arr, ArrayLike):
        raise TypeError(
            f"Expected {err_name} to be arraylike; got {type(arr).__name__}.",
        )
    return jnp.asarray(arr, **kwargs)


def _infer_axis_size_from_params(tree: PyTree, in_axes) -> int:
    axes = _resolve_vmapped_axes(tree, in_axes)
    axis_sizes = tree_leaves(
        tree_map(
            lambda leaf, ax: leaf.shape[ax] if ax is not None else None,
            tree,
            axes,
        ),
    )
    if len(axis_sizes) == 0:
        raise ValueError("in_axes did not map to any leaves to vectorize.")
    return axis_sizes[0]


def _resolve_vmapped_axes(pytree, in_axes):
    """Returns pytree with ints denoting vmapped dimensions."""

    # Adapted from equinox filter_vmap
    def _resolve_axis(in_axes, elem):
        if in_axes is None or isinstance(in_axes, int):
            return tree_map(lambda _: in_axes, elem)
        if callable(in_axes):
            return tree_map(in_axes, elem)
        raise TypeError("`in_axes` must consist of None, ints, and callables.")

    return tree_map(_resolve_axis, in_axes, pytree, is_leaf=lambda x: x is None)
