from flowjax.bijections import Bijection
import jax.numpy as jnp
from typing import Sequence
from flowjax.utils import merge_shapes


class Concatenate(Bijection):
    split_idxs: tuple[int]
    bijections: Sequence[Bijection]
    axis: int

    def __init__(self, bijections: Sequence[Bijection], axis: int = 0):

        # TODO what if len(bijections)==1? What does concatenate do?
        # TODO test -1
        self.bijections = bijections
        self.axis = axis

        shapes = [b.shape for b in bijections]
        self._check_shapes(shapes, axis)
        self.shape = (*shapes[0][:axis], sum(s[axis] for s in shapes), *shapes[0][axis+1:])    
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
    
    @staticmethod
    def _check_shapes(shapes, axis):
        if any(s is None for s in shapes):
            raise ValueError(
                "Cannot concatenate bijections with shape None. You may wish to "
                "explicitly set shape during initialisation."
            )
        
        expected_dim = len(shapes[0])
        expected_matching = (*shapes[:axis], *shapes[axis+1:])

        for i, shape in enumerate(shapes):
            if len(shape) != expected_dim:
                raise ValueError(
                    "All bijections must have the same number of dimensions, but "
                    f"the bijection at index 0 had {expected_dim} dimensions(s) "
                    f"and the bijection at index {i} had {len(shape)} dimension(s). "
                    )
            
            elif (*shapes[:axis], *shapes[axis+1:]) != expected_matching:
                raise ValueError(
                    "All bijection dimensions must match, except along dimension "
                    "corresponding to axis, but the bijection at index 0 had shape "
                    f"{shapes[0]}, and the bijection at index {i} had {len(shape)} "
                    "dimension(s)."
                    )
        

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
        self.shape = (*shapes[0][:axis], len(bijections), *shapes[0][axis:])   
        self.cond_shape = merge_shapes([b.cond_shape for b in bijections]) 

        
        # TODO check shape mismatches
        # TODO what if len(bijections)==1? What does concatenate do?
        # TODO test -1
        # TODO conditional checks?s


    def transform(self, x, condition = None):
        xs = jnp.split(x, len(self.bijections), axis=self.axis)
        ys = [b.transform(x, condition) for (b, x) in zip(self.bijections, xs)]
        return jnp.stack(ys, self.axis)
    
    def transform_and_log_abs_det_jacobian(self, x, condition = None):
        xs = jnp.split(x, len(self.bijections), axis=self.axis)

        ys_log_det = [
            b.transform_and_log_abs_det_jacobian(x, condition)
            for b, x in zip(self.bijections, xs)
            ]
        
        ys, log_dets = zip(*ys_log_det)
        return jnp.stack(ys, self.axis), log_dets
        
    def inverse(self, y, condition = None):
        ys = jnp.split(y, len(self.bijections), axis=self.axis)
        xs = [b.inverse(y, condition) for (b, y) in zip(self.bijections, ys)]
        return jnp.stack(xs, self.axis)

    def inverse_and_log_abs_det_jacobian(self, y, condition = None):
        ys = jnp.split(y, len(self.bijections), axis=self.axis)
        xs_log_det = [
            b.inverse_and_log_abs_det_jacobian(y, condition)
            for b, y in zip(self.bijections, ys)
            ]
        xs, log_dets = zip(*xs_log_det)
        return jnp.stack(xs, self.axis), log_dets
    