"""
Abstact base classes for the `Bijection` and `Transformer` types. Note when implementing bijections,
by convention we try to i) implement the "transform" methods as the faster/more intuitive approach 
(compared to the inverse methods); and ii) implement only the forward methods if an inverse
is not available. The `Invert` bijection can be used to invert the orientation if a fast inverse is
desired (e.g. maximum likelihood fitting of flows).
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from equinox import Module

from flowjax.utils import Array
import warnings

from flowjax.transformers import Transformer as _Transformer

    
class Transformer(_Transformer):
    def __init__(self, *args, **kwargs) -> None:
        "Deprecated location of transfomer, use flowjax.transformers.Transformer instead."
        warnings.warn(
            "Please use flowjax.transformers.Transformer.",
            DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)
    

class Bijection(ABC, Module):
    """Basic bijection class. All bijections should support conditioning variables
    (even if ignored)."""

    cond_dim: int

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

