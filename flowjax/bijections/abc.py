from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import Optional


class Bijection(ABC):
    """Basic bijection class. All bijections should support conditioning variables
    (even if ignored)."""
    cond_dim: int

    @abstractmethod
    def transform(self, x: jnp.ndarray, condition: Optional[jnp.ndarray] = None):
        """Apply transformation."""
        pass

    @abstractmethod
    def transform_and_log_abs_det_jacobian(
        self, x: jnp.ndarray, condition: Optional[jnp.ndarray] = None
    ):
        """Apply transformation and compute log absolute value of the Jacobian determinant."""
        pass

    @abstractmethod
    def inverse(self, y: jnp.ndarray, condition: Optional[jnp.ndarray] = None):
        """Invert the transformation."""
        pass


class ParameterisedBijection(ABC):
    """Bijection which facilitates parameterisation with a
    neural network output (e.g. as in coupling flows)."""

    @abstractmethod
    def transform(self, x: jnp.ndarray, *args):
        """Apply transformation."""
        pass

    @abstractmethod
    def transform_and_log_abs_det_jacobian(self, x: jnp.ndarray, *args):
        """Apply transformation and compute log absolute value of the Jacobian determinant."""
        pass

    @abstractmethod
    def inverse(self, y: jnp.ndarray, *args):
        """Invert the transformation."""
        pass

    @abstractmethod
    def num_params(self, dim: int):
        "Total number of parameters required for bijection."
        pass

    @abstractmethod
    def get_args(self, params: jnp.ndarray):
        "Transform unconstrained vector of params (e.g. nn output) into args for transformation."
        pass
