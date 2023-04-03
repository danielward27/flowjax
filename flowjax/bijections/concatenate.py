from flowjax.bijections import Bijection
import jax.numpy as jnp
from typing import Sequence
from flowjax.utils import merge_shapes
from jax import Array

class Concatenate(Bijection):
    split_idxs: tuple[int]
    bijections: Sequence[Bijection]
    axis: int

    def __init__(self, bijections: Sequence[Bijection], axis: int = 0):
        self.bijections = bijections
        self.axis = axis


        axis = range(len(shapes[0]))[axis]  # allows negative axis specification
        shapes = [b.shape for b in bijections]
        self.shape = self._infer_shape(shapes)  
        self.split_idxs = [s[axis] for s in shapes[:-1]]
        self.cond_shape = merge_shapes([b.cond_shape for b in bijections])
        
    def transform(self, x, condition = None):
        x_parts = jnp.array_split(x, self.split_idxs, axis=self.axis)
        ys = [b.transform(x_part, condition) for b, x_part in zip(self.bijections, x_parts)]
        return jnp.concatenate(ys, axis=self.axis)
    
    def transform_and_log_abs_det_jacobian(self, x, condition = None):
        xs = jnp.array_split(x, self.split_idxs, axis=self.axis)
        
        ys_log_dets = [
            b.transform_and_log_abs_det_jacobian(x, condition)
            for b, x in zip(self.bijections, xs)
        ]

        ys, log_dets = zip(*ys_log_dets)
        return jnp.concatenate(ys, self.axis), sum(log_dets)
            
    def inverse(self, y, condition = None):
        y_parts = jnp.array_split(y, self.split_idxs, axis=self.axis)
        xs = [b.inverse(y_part, condition) for b, y_part in zip(self.bijections, y_parts)]
        return jnp.concatenate(xs, axis=self.axis)
    
    def inverse_and_log_abs_det_jacobian(self, y, condition = None):
        ys = jnp.array_split(y, self.split_idxs, axis=self.axis)

        xs_log_dets = [
            b.inverse_and_log_abs_det_jacobian(y, condition)
            for b, y in zip(self.bijections, ys)
        ]

        xs, log_dets = zip(*xs_log_dets)
        return jnp.concatenate(xs, self.axis), sum(log_dets) 
    
    def _infer_shape(self, shapes):
        if any(s is None for s in shapes):
            raise ValueError(
                "Cannot concatenate bijections with shape None. You may wish to "
                "explicitly set shape during initialisation."
            )
        dim = len(shapes[0])
        try:
            axis = range(dim)[self.axis]
        except IndexError:
            raise IndexError(f"Invalid axis {self.axis} for {dim}-dimensional bijection")
        
        for i, shape in enumerate(shapes):
            if len(shape) != dim:
                raise ValueError(
                    f"Bijections must have consistent number of dimensions, but index "
                    f"0 has {dim} dimensions and index {i} has {len(shape)}."
                    )
            
            if (*shape[:axis], *shape[axis+1:]) != (shapes[0][:axis], shapes[0][axis+1:]):
                raise ValueError(
                    f"Expected bijection shapes to match except along axis {axis}, but "
                    f"index 0 had shape {shapes[0]}, and index {i} had shape {shape}."
                    )
        
        return (*shapes[0][:axis], sum(s[axis] for s in shapes), *shapes[0][axis+1:])   

        
        

class Stack(Bijection):
    bijections: list[Bijection]
    axis: int

    def __init__(self, bijections: list[Bijection], axis: int = 0):
        """Stack bijections along a new axis (analagous to jnp.stack).

        Args:
            bijections (list[Bijection]): Bijections.
            axis (int, optional): Axis along which to stack. Defaults to 0.
        """
        self.axis = axis
        self.bijections = bijections

        shapes = [b.shape for b in bijections]
        if not all([s==shapes[0] for s in shapes]):
            raise ValueError(
                "All input bijections must have the same shape."
            )
        self.shape = (*shapes[0][:axis], len(bijections), *shapes[0][axis:])   
        self.cond_shape = merge_shapes([b.cond_shape for b in bijections]) 

        # TODO check shape mismatches
        # TODO conditional checks?


    def transform(self, x, condition = None):
        xs = self._split_and_squeeze(x)
        ys = [
            b.transform(x, condition)
            for (b, x) in zip(self.bijections, xs)
            ]
        return jnp.stack(ys, self.axis)
    
    def transform_and_log_abs_det_jacobian(self, x, condition = None):
        xs = self._split_and_squeeze(x)
        ys_log_det = [
            b.transform_and_log_abs_det_jacobian(x, condition)
            for b, x in zip(self.bijections, xs)
            ]
        
        ys, log_dets = zip(*ys_log_det)
        return jnp.stack(ys, self.axis), sum(log_dets)
        
    def inverse(self, y, condition = None):
        ys = self._split_and_squeeze(y)
        xs = [b.inverse(y, condition) for (b, y) in zip(self.bijections, ys)]
        return jnp.stack(xs, self.axis)

    def inverse_and_log_abs_det_jacobian(self, y, condition = None):
        ys = self._split_and_squeeze(y)
        xs_log_det = [
            b.inverse_and_log_abs_det_jacobian(y, condition)
            for b, y in zip(self.bijections, ys)
            ]
        xs, log_dets = zip(*xs_log_det)
        return jnp.stack(xs, self.axis), sum(log_dets)

    def _split_and_squeeze(self, array: Array):
        arrays = jnp.split(array, len(self.bijections), axis=self.axis)
        return [a.squeeze(axis=self.axis) for a in arrays]
