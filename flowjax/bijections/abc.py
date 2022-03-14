from abc import ABC, abstractmethod


class Bijection(ABC):
    "Basic bijection class"

    @abstractmethod
    def transform(self, x, *args):
        pass

    @abstractmethod
    def transform_and_log_abs_det_jacobian(self, x, *args):
        pass

    @abstractmethod
    def inverse(self, y, *args):
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
