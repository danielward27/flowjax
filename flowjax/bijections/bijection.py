"""
Abstact base classes for the `Bijection` and `_Transformer` types. Note when implementing bijections,
by convention we try to i) implement the "transform" methods as the faster/more intuitive approach 
(compared to the inverse methods); and ii) implement only the forward methods if an inverse
is not available. The `Invert` bijection can be used to invert the orientation if a fast inverse is
desired (e.g. maximum likelihood fitting of flows).
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

from equinox import Module

from flowjax.utils import Array

class Bijection(ABC, Module):
    """Basic bijection class. All bijections support conditioning variables
    (even if ignored).
    
    A shape of None can be used to represent compatibility with any shape input.
    An element of a shape -1 can be used to represent compatibility with any length on
    that axis. For example, (-1, ) would represent any vector.    
    
    Attributes:
        shape (Union[None, Tuple[int]]): The shape of the variable to be transformed.
        cond_shape (Union[None, Tuple[int]]): The shape of the conditioning variables.
    """
    shape: Union[None, Tuple[int]]
    cond_shape: Union[None, Tuple[int]]

    @abstractmethod
    def transform(self, x: Array, condition: Optional[Array] = None) -> Array:
        """Apply transformation."""

    @abstractmethod
    def transform_and_log_abs_det_jacobian(
        self, x: Array, condition: Optional[Array] = None
    ) -> Tuple:
        """Apply transformation and compute log absolute value of the Jacobian determinant."""

    @abstractmethod
    def inverse(self, y: Array, condition: Optional[Array] = None) -> Array:
        """Invert the transformation."""

    @abstractmethod
    def inverse_and_log_abs_det_jacobian(
        self, y: Array, condition: Optional[Array] = None
    ) -> Array:
        """Invert the transformation and compute log absolute value of the Jacobian determinant."""

    def _argcheck(self, x, condition=None):
        "Utility argcheck that can be added to bijection methods as required."
        if x.shape != self.shape:
            raise ValueError(f"Expected x.shape {self.shape}, got {x.shape}")

        if self.cond_shape is not None:
            if condition is None:
                raise ValueError("Condition should be provided")
            elif condition.shape != self.cond_shape:
                raise ValueError(f"Expected condition.shape {self.cond_shape}, got {condition.shape}")


