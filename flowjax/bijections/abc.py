from abc import ABC, abstractmethod
import jax.numpy as jnp


class Bijection(ABC):
    """Basic bijection class. All bijections should support conditioning variables
    (even if ignored)."""

    @abstractmethod
    def transform(self, x: jnp.ndarray, *args, condition: jnp.ndarray = jnp.array([])):
        """Apply transformation."""
        pass

    @abstractmethod
    def transform_and_log_abs_det_jacobian(self, x, *args, condition=jnp.array([])):
        """Apply transformation and compute log absolute value of the Jacobian determinant."""
        pass

    @abstractmethod
    def inverse(self, y, *args, condition=jnp.array([])):
        """Invert the transformation."""
        pass


class ParameterisedBijection(Bijection, ABC):
    """Bijection with additional methods facilitating parameterisation with a
    neural network."""

    @abstractmethod
    def num_params(self, dim: int):
        "Total number of parameters required for bijection."
        pass

    @abstractmethod
    def get_args(self, params):
        "Transform unconstrained vector of params (e.g. nn output) into args for transformation."
        pass
