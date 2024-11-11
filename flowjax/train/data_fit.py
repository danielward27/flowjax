"""Function to fit flows to samples from a distribution."""

import warnings

from flowjax import train


def fit_to_data(*args, **kwargs):  # TODO deprecate
    """Deprecated import for fit_to_data."""
    warnings.warn(
        "Importing from data_fit will be deprecated in 17.0.0.. Please import from "
        "``flowjax.train.loops`` or ``flowjax.train``.",
        DeprecationWarning,
        stacklevel=2,
    )
    return train.loops.fit_to_data(*args, **kwargs)
