"""Function to fit flows to samples from a distribution."""
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import Array
from jax.typing import ArrayLike
from tqdm import tqdm

from flowjax.distributions import AbstractDistribution
from flowjax.train.losses import MaximumLikelihoodLoss
from flowjax.train.train_utils import (
    count_fruitless,
    get_batches,
    step,
    train_val_split,
)

PyTree = Any


def fit_to_data(
    key: Array,
    dist: AbstractDistribution,
    x: ArrayLike,
    condition: ArrayLike = None,
    loss_fn: Callable | None = None,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    val_prop: float = 0.1,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    filter_spec: Callable | PyTree = eqx.is_inexact_array,
    show_progress: bool = True,
):
    r"""Train a distribution (e.g. a flow) to samples from the target distribution.

    The distribution can be unconditional :math:`p(x)` or conditional
    :math:`p(x|\text{condition})`. Note that the last batch in each epoch is dropped
    if truncated.

    Args:
        key (KeyArray): Jax random seed.
        dist (AbstractDistribution): Distribution object.
        x (ArrayLike): Samples from target distribution.
        condition (ArrayLike | None): Conditioning variables. Defaults to None.
        loss_fn (Callable | None): Loss function. Defaults to MaximumLikelihoodLoss.
        max_epochs (int): Maximum number of epochs. Defaults to 100.
        max_patience (int): Number of consecutive epochs with no validation
            loss improvement after which training is terminated. Defaults to 5.
        batch_size (int): Batch size. Defaults to 100.
        val_prop (float): Proportion of data to use in validation set. Defaults to 0.1.
        learning_rate (float): Adam learning rate. Defaults to 5e-4.
        optimizer (optax.GradientTransformation): Optax optimizer. If provided, this
            overrides the default Adam optimizer, and the learning_rate is ignored.
            Defaults to None.
        filter_spec (Callable | PyTree): Equinox `filter_spec` for specifying trainable
            parameters. Either a callable `leaf -> bool`, or a PyTree with prefix
            structure matching `dist` with True/False values. Defaults to
            `eqx.is_inexact_array`.
        show_progress (bool): Whether to show progress bar. Defaults to True.
    """
    data = (x,) if condition is None else (x, condition)
    data = tuple(jnp.asarray(a) for a in data)

    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    if loss_fn is None:
        loss_fn = MaximumLikelihoodLoss()

    params, static = eqx.partition(dist, filter_spec)
    best_params = params
    opt_state = optimizer.init(params)

    # train val split
    key, subkey = jr.split(key)
    train_data, val_data = train_val_split(key, data, val_prop=val_prop)
    losses = {"train": [], "val": []}

    loop = tqdm(range(max_epochs), disable=not show_progress)

    for _ in loop:
        # Shuffle data
        key, *subkeys = jr.split(key, 3)
        train_data = [jr.permutation(subkeys[0], a) for a in train_data]
        val_data = [jr.permutation(subkeys[1], a) for a in val_data]

        # Train epoch
        batch_losses = []
        for batch in zip(*get_batches(train_data, batch_size), strict=True):
            params, opt_state, loss_i = step(
                optimizer,
                opt_state,
                loss_fn,
                params,
                static,
                *batch,
            )
            batch_losses.append(loss_i)
        losses["train"].append(sum(batch_losses) / len(batch_losses))

        # Val epoch
        batch_losses = []
        for batch in zip(*get_batches(val_data, batch_size), strict=True):
            loss_i = loss_fn(params, static, *batch)
            batch_losses.append(loss_i)
        losses["val"].append(sum(batch_losses) / len(batch_losses))

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = params

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    dist = eqx.combine(best_params, static)
    return dist, losses
