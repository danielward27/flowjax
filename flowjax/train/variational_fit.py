"""Basic training script for fitting a flow using variational inference."""
from typing import Any, Callable

import equinox as eqx
import jax.random as jr
import optax
from tqdm import tqdm

from flowjax.distributions import Distribution
from flowjax.train.losses import Loss
from flowjax.train.train_utils import step

PyTree = Any


def fit_to_variational_target(
    key: jr.KeyArray,
    dist: Distribution,
    loss_fn: Loss,
    steps: int = 100,
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
        loss_fn (Loss | None): The loss function to optimize (e.g. the ElboLoss).
        steps (int, optional): The number of training steps to run. Defaults to 100.
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

    @eqx.filter_value_and_grad
    def loss_val_and_grad(trainable, static, key):
        model = eqx.combine(trainable, static)
        return loss_fn(model, key)

    params, static = eqx.partition(dist, filter_spec)
    opt_state = optimizer.init(params)

    losses = []

    best_params = params
    keys = tqdm(jr.split(key, steps), disable=not show_progress)

    for key in keys:
        params, opt_state, loss = step(
            optimizer, opt_state, loss_fn, params, static, key
        )
        losses.append(loss.item())
        keys.set_postfix({"loss": loss.item()})
        if loss.item() == min(losses):
            best_params = params

    return eqx.combine(best_params, static), losses
