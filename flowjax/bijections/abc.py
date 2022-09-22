"""
Abstact base classes for the `Bijection` and `Transformer` types. Note when implementing bijections,
by convention we try to i) implement the "transform" methods as the faster/more intuitive approach 
(compared to the inverse methods); and ii) implement only the forward methods if an inverse
is not available. The `Invert` bijection can be used to invert the orientation if a fast inverse is
desired (e.g. maximum likelihood fitting of flows).
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

    @abstractmethod
    def transform_and_log_abs_det_jacobian(
        self, x: Array, condition: Optional[Array] = None
    ) -> Tuple:
        """Apply transformation and compute log absolute value of the Jacobian determinant."""

    @abstractmethod
    def inverse(self, y: Array, condition: Optional[Array] = None) -> Array:
        """Invert the transformation."""

    @abstractmethod
    def inverse_and_log_abs_det_jacobian(self, y: Array, condition: Optional[Array] = None) -> Array:
        """Invert the transformation and compute log absolute value of the Jacobian determinant."""


class Transformer(ABC):
    """Bijection which facilitates parameterisation with a neural network output
    (e.g. as in coupling flows, or masked autoressive flows). Should not contain
    (directly) trainable parameters."""

    @abstractmethod
    def transform(self, x: Array, *args: Array) -> Array:
        """Apply transformation."""

    @abstractmethod
    def transform_and_log_abs_det_jacobian(self, x: Array, *args: Array) -> tuple:
        """Apply transformation and compute log absolute value of the Jacobian determinant."""

    @abstractmethod
    def inverse(self, y: Array, *args: Array) -> Array:
        """Invert the transformation."""

    def inverse_and_log_abs_det_jacobian(self, x: Array, *args: Array) -> tuple:
        """Invert the transformation and compute the log absolute value of the Jacobian determinant."""

    @abstractmethod
    def num_params(self, dim: int) -> int:
        "Total number of parameters required for bijection."

    @abstractmethod
    def get_ranks(self, dim: int) -> Array:
        "The ranks of the parameters, i.e. which dimension of the input the parameters correspond to."

    @abstractmethod
    def get_args(self, params: Array) -> List[Array]:
        "Transform unconstrained vector of params (e.g. nn output) into args for transformation."
