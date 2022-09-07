"""
Abstact base classes for the `Bijection` and `ParameterisedBijection` types
"""

from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import List, Optional, Tuple
from equinox import Module
from flowjax.utils import Array


class Bijection(ABC, Module):
    """Basic bijection class. All bijections should support conditioning variables
    (even if ignored)."""

    cond_dim: int

    @abstractmethod
    def transform(self, x: Array, condition: Optional[Array] = None) -> Array:
        """Apply transformation."""
        pass

    @abstractmethod
    def transform_and_log_abs_det_jacobian(
        self, x: Array, condition: Optional[Array] = None
    ) -> Tuple:
        """Apply transformation and compute log absolute value of the Jacobian determinant."""
        pass

    @abstractmethod
    def inverse(self, y: Array, condition: Optional[Array] = None) -> Array:
        """Invert the transformation."""
        pass


class ParameterisedBijection(ABC):
    """Bijection which facilitates parameterisation with a neural network output
    (e.g. as in coupling flows, or masked autoressive flows)."""

    @abstractmethod
    def transform(self, x: Array, *args: Array) -> Array:
        """Apply transformation."""
        pass

    @abstractmethod
    def transform_and_log_abs_det_jacobian(self, x: Array, *args: Array) -> tuple:
        """Apply transformation and compute log absolute value of the Jacobian determinant."""
        pass

    @abstractmethod
    def inverse(self, y: Array, *args: Array) -> Array:
        """Invert the transformation."""
        pass

    @abstractmethod
    def num_params(self, dim: int) -> int:
        "Total number of parameters required for bijection."
        pass

    @abstractmethod
    def get_ranks(self, dim: int) -> Array:
        "The ranks of the parameters, i.e. which dimension of the input the parameters correspond to."
        pass

    @abstractmethod
    def get_args(self, params: Array) -> List[Array]:
        "Transform unconstrained vector of params (e.g. nn output) into args for transformation."
        pass
