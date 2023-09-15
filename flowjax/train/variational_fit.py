"""Basic training script for fitting a flow using variational inference."""
from typing import Any, Callable

import equinox as eqx
import jax.random as jr
import optax
from jax.typing import ArrayLike
from tqdm import tqdm

from flowjax.distributions import Distribution
from flowjax.train.train_utils import step

PyTree = Any


def fit_to_variational_target(
    key: jr.KeyArray,
    dist: Distribution,
    loss_fn: Callable,
    steps: int = 100,
    condition: ArrayLike | None = None,
    batch_size: int = 100,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    filter_spec: Callable | PyTree = eqx.is_inexact_array,
    show_progress: bool = True,
):
    """Train a distribution (e.g. a flow) by variational inference to a target
    (e.g. an unnormalized density).

    Args:
        key (jr.KeyArray): Jax PRNGKey.
        dist (Distribution): Distribution object, trainable parameters are found
            using equinox.is_inexact_array.
        loss_fn (Callable | None): The loss function to optimize (e.g. the ElboLoss).
        steps (int, optional): The number of training steps to run. Defaults to 100.
        condition (ArrayLike | None): Conditioning variables for learning an amortized
            conditional distribution (with a batch axis on axis 0). For example, this
            allows using a single density estimator to learn the posterior distribution
            for multiple observations. Note that no cross validation is currently used,
            so care should be used if attempting to generalise to new observations.
        batch_size (int): Batch size for conditioning variables. Ignored if condition
            is None.
        learning_rate (float, optional): Learning rate. Defaults to 5e-4.
        optimizer (optax.GradientTransformation | None, optional): Optax optimizer. If
            provided, this overrides the default Adam optimizer, and the learning_rate
            is ignored. Defaults to None.
        filter_spec (Callable | PyTree, optional): Equinox `filter_spec` for
            specifying trainable parameters. Either a callable `leaf -> bool`, or a
            PyTree with prefix structure matching `dist` with True/False values.
            Defaults to eqx.is_inexact_array.
        show_progress (bool, optional): Whether to show progress bar. Defaults to True.

    Returns:
        tuple: (distribution, losses).
    """
    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    params, static = eqx.partition(dist, filter_spec)
    opt_state = optimizer.init(params)

    losses = []

    best_params = params
    keys = tqdm(jr.split(key, steps), disable=not show_progress)

    for key in keys:
        if condition is not None:
            key, subkey = jr.split(key)
            cond_batch = jr.choice(subkey, condition, (batch_size,))

        loss_args = (key,) if condition is None else (key, cond_batch)

        params, opt_state, loss = step(
            optimizer, opt_state, loss_fn, params, static, *loss_args
        )
        losses.append(loss.item())
        keys.set_postfix({"loss": loss.item()})
        if loss.item() == min(losses):
            best_params = params

    return eqx.combine(best_params, static), losses
