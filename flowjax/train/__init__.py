"""The train sub-package contains basic functions for training flows, to
samples from a target distribution, or by using variational inference.
"""

from .data_fit import fit_to_data
from .variational_fit import fit_to_variational_target

__all__ = [
    "fit_to_data",
    "fit_to_variational_target",
]
