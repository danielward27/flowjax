"""Module contains bijections formed by "stacking/concatenating" other bijections."""

from typing import Sequence

import jax.numpy as jnp
from jax import Array

from flowjax.bijections.bijection import Bijection
from flowjax.utils import merge_cond_shapes, check_shapes_match


class Concatenate(Bijection):
    """Concatenate bijections along an already existing axis. Analagous to
    ``jnp.concatenate``. See also :class:`Stack`.
    """

    split_idxs: Array
    bijections: Sequence[Bijection]
    axis: int

    def __init__(self, bijections: Sequence[Bijection], axis: int = 0):
        """
        Args:
            bijections (Sequence[Bijection]): Bijections, to stack into a single
                bijection.
            axis (int): Axis along which to stack. Defaults to 0.
        """
        self.bijections = bijections
        self.axis = axis

        shapes = [b.shape for b in bijections]
        self._argcheck_shapes(shapes)
        axis = range(len(shapes[0]))[axis]  # Avoids issues when axis==-1
        self.shape = (
            shapes[0][:axis] + (sum(s[axis] for s in shapes),) + shapes[0][axis + 1 :]
        )
        self.split_idxs = jnp.array([s[axis] for s in shapes[:-1]])
        self.cond_shape = merge_cond_shapes([b.cond_shape for b in bijections])

    def transform(self, x, condition=None):
        self._argcheck(x, condition)
        x_parts = jnp.array_split(x, self.split_idxs, axis=self.axis)
        y_parts = [
            b.transform(x_part, condition)
            for b, x_part in zip(self.bijections, x_parts)
        ]
        return jnp.concatenate(y_parts, axis=self.axis)

    def transform_and_log_det(self, x, condition=None):
        self._argcheck(x, condition)
        x_parts = jnp.array_split(x, self.split_idxs, axis=self.axis)

        ys_log_dets = [
            b.transform_and_log_det(x, condition)
            for b, x in zip(self.bijections, x_parts)
        ]

        y_parts, log_dets = zip(*ys_log_dets)
        return jnp.concatenate(y_parts, self.axis), sum(log_dets)

    def inverse(self, y, condition=None):
        self._argcheck(y, condition)
        y_parts = jnp.array_split(y, self.split_idxs, axis=self.axis)
        x_parts = [
            b.inverse(y_part, condition) for b, y_part in zip(self.bijections, y_parts)
        ]
        return jnp.concatenate(x_parts, axis=self.axis)

    def inverse_and_log_det(self, y, condition=None):
        self._argcheck(y, condition)
        y_parts = jnp.array_split(y, self.split_idxs, axis=self.axis)

        xs_log_dets = [
            b.inverse_and_log_det(y, condition)
            for b, y in zip(self.bijections, y_parts)
        ]

        x_parts, log_dets = zip(*xs_log_dets)
        return jnp.concatenate(x_parts, self.axis), sum(log_dets)

    def _argcheck_shapes(self, shapes: list[tuple[int, ...]]):
        axis = range(len(shapes[0]))[self.axis]

        for i, shp in enumerate(shapes):
            if shp[:axis] + shp[axis + 1 :] != shapes[0][:axis] + shapes[0][axis + 1 :]:
                raise ValueError(
                    f"Expected bijection shapes to match except along axis {axis}, but "
                    f"index 0 had shape {shapes[0]}, and index {i} had shape {shp}."
                )


class Stack(Bijection):
    """
    Stack bijections along a new axis (analagous to ``jnp.stack``).
    See also :class:`Concatenate`.
    """

    bijections: Sequence[Bijection]
    axis: int

    def __init__(self, bijections: list[Bijection], axis: int = 0):
        """
        Args:
            bijections (list[Bijection]): Bijections.
            axis (int): Axis along which to stack. Defaults to 0.
        """
        self.axis = axis
        self.bijections = bijections

        shapes = [b.shape for b in bijections]
        check_shapes_match(shapes)

        self.shape = shapes[0][:axis] + (len(bijections),) + shapes[0][axis:]
        self.cond_shape = merge_cond_shapes([b.cond_shape for b in bijections])

    def transform(self, x, condition=None):
        self._argcheck(x, condition)
        x_parts = self._split_and_squeeze(x)
        y_parts = [
            b.transform(x, condition) for (b, x) in zip(self.bijections, x_parts)
        ]
        return jnp.stack(y_parts, self.axis)

    def transform_and_log_det(self, x, condition=None):
        self._argcheck(x, condition)
        x_parts = self._split_and_squeeze(x)
        ys_log_det = [
            b.transform_and_log_det(x, condition)
            for b, x in zip(self.bijections, x_parts)
        ]

        y_parts, log_dets = zip(*ys_log_det)
        return jnp.stack(y_parts, self.axis), sum(log_dets)

    def inverse(self, y, condition=None):
        self._argcheck(y, condition)
        y_parts = self._split_and_squeeze(y)
        x_parts = [b.inverse(y, condition) for (b, y) in zip(self.bijections, y_parts)]
        return jnp.stack(x_parts, self.axis)

    def inverse_and_log_det(self, y, condition=None):
        self._argcheck(y, condition)
        y_parts = self._split_and_squeeze(y)
        xs_log_det = [
            b.inverse_and_log_det(y, condition)
            for b, y in zip(self.bijections, y_parts)
        ]
        x_parts, log_dets = zip(*xs_log_det)
        return jnp.stack(x_parts, self.axis), sum(log_dets)

    def _split_and_squeeze(self, array: Array):
        arrays = jnp.split(array, len(self.bijections), axis=self.axis)
        return [a.squeeze(axis=self.axis) for a in arrays]
