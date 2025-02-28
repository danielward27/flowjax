"""Training loops."""

import warnings
from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.random as jr
import optax
import paramax
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree, Scalar
from tqdm import tqdm

from flowjax.train.losses import MaximumLikelihoodLoss
from flowjax.train.train_utils import (
    count_fruitless,
    get_batches,
    step,
    train_val_split,
)
from flowjax.utils import arraylike_to_array


def fit_to_key_based_loss(
    key: PRNGKeyArray,
    tree: PyTree,
    *,
    loss_fn: Callable[[PyTree, PyTree, PRNGKeyArray], Scalar],
    steps: int,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    show_progress: bool = True,
):
    """Train a pytree, using a loss with params, static and key as arguments.

    This can be used e.g. to fit a distribution using a variational objective, such as
    the evidence lower bound.

    Args:
        key: Jax random key.
        tree: PyTree, from which trainable parameters are found using
            ``equinox.is_inexact_array``.
        loss_fn: The loss function to optimize.
        steps: The number of optimization steps.
        learning_rate: The adam learning rate. Ignored if optimizer is provided.
        optimizer: Optax optimizer. Defaults to None.
        show_progress: Whether to show progress bar. Defaults to True.

    Returns:
        A tuple containing the trained pytree and the losses.
    """
    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    params, static = eqx.partition(
        tree,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    opt_state = optimizer.init(params)

    losses = []

    keys = tqdm(jr.split(key, steps), disable=not show_progress)

    for key in keys:
        params, opt_state, loss = step(
            params,
            static,
            key=key,
            optimizer=optimizer,
            opt_state=opt_state,
            loss_fn=loss_fn,
        )
        losses.append(loss.item())
        keys.set_postfix({"loss": loss.item()})
    return eqx.combine(params, static), losses


def fit_to_data(
    key: PRNGKeyArray,
    dist: PyTree,  # Custom losses may support broader types than AbstractDistribution
    *data: ArrayLike,
    loss_fn: Callable | None = None,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    val_prop: float = 0.1,
    return_best: bool = True,
    show_progress: bool = True,
    x: ArrayLike | None = None,
    condition: ArrayLike | None = None,
):
    r"""Train a PyTree (e.g. a distribution) to samples from the target.

    The model can be unconditional :math:`p(x)` or conditional
    :math:`p(x|\text{condition})`. Note that the last batch in each epoch is dropped
    if truncated (to avoid recompilation). This function can also be used to fit
    non-distribution pytrees as long as a compatible loss function is provided.

    Args:
        key: Jax random seed.
        dist: The pytree to train (usually a distribution).
        *data: A variable number of data arrays with matching shape on axis 0. Batches
            of each array are passed to the loss function as positional arguments
            (see documentation for ``loss_fn``). Commonly this is a single array for
            unconditional density estimation, or two arrays ``*(target, condition)``
            for conditional density estimation.
        learning_rate: The learning rate for adam optimizer. Ignored if optimizer is
            provided.
        optimizer: Optax optimizer. Defaults to None.
        loss_fn: Loss function. The signature should be of the form
            ``(params, static, *arrays, key)``. Defaults to MaximumLikelihoodLoss.
        max_epochs: Maximum number of epochs. Defaults to 100.
        max_patience: Number of consecutive epochs with no validation loss improvement
            after which training is terminated. Defaults to 5.
        batch_size: Batch size. Defaults to 100.
        val_prop: Proportion of data to use in validation set. Defaults to 0.1.
        return_best: Whether the result should use the parameters where the minimum loss
            was reached (when True), or the parameters after the last update (when
            False). Defaults to True.
        show_progress: Whether to show progress bar. Defaults to True.
        x: Deprecated. Pass in as positional argument instead. See variable argument
            *data.
        condition: Deprecated way to pass conditioning variables. Pass as a positional
            argument instead. See variable argument *data.

    Returns:
        A tuple containing the trained distribution and the losses.
    """

    def _handle_deprecation(data, x, condition):
        # TODO This function handles the deprecation of x and condition, so will
        # be removed when deprecated.

        if data != () and x is not None:  # Note x passed as key word in this case
            raise ValueError("Use data argument only (pass x in data).")

        if x is not None:
            warnings.warn(
                "Argument x to fit_to_data is deprecated and will be removed in the "
                "next major version. Pass as a positional argument instead. See "
                "documentation of *data. This change allows for more flexibility in "
                "the number of arrays required by a loss.",
                DeprecationWarning,
                stacklevel=2,
            )
            data += (x,)

        if condition is not None:
            warnings.warn(
                "condition is deprecated and will be removed in the next major "
                "version. Pass both x and condition as positional arguments. "
                "See documentation of *data. This change allows for more flexibility "
                "in the number of arrays required by a loss.",
                DeprecationWarning,
                stacklevel=2,
            )
            data += (condition,)

        return data

    data = _handle_deprecation(data, x, condition)

    data = jax.tree.map(arraylike_to_array, data)

    if loss_fn is None:
        loss_fn = MaximumLikelihoodLoss()

    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    params, static = eqx.partition(
        dist,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    best_params = params
    opt_state = optimizer.init(params)

    # train val split
    key, subkey = jr.split(key)
    train_data, val_data = train_val_split(subkey, data, val_prop=val_prop)
    losses = {"train": [], "val": []}

    loop = tqdm(range(max_epochs), disable=not show_progress)

    for _ in loop:
        # Shuffle data
        key, *subkeys = jr.split(key, 3)
        train_data = jax.tree.map(partial(jr.permutation, subkeys[0]), train_data)
        val_data = jax.tree.map(partial(jr.permutation, subkeys[1]), val_data)

        # Train epoch
        batch_losses = []
        for batch in zip(*get_batches(train_data, batch_size), strict=True):
            key, subkey = jr.split(key)
            params, opt_state, loss_i = step(
                params,
                static,
                *batch,
                optimizer=optimizer,
                opt_state=opt_state,
                loss_fn=loss_fn,
                key=subkey,
            )
            batch_losses.append(loss_i)
        losses["train"].append((sum(batch_losses) / len(batch_losses)).item())

        # Val epoch
        batch_losses = []
        for batch in zip(*get_batches(val_data, batch_size), strict=True):
            key, subkey = jr.split(key)
            loss_i = loss_fn(params, static, *batch, key=subkey)
            batch_losses.append(loss_i)
        losses["val"].append((sum(batch_losses) / len(batch_losses)).item())

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = params

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    params = best_params if return_best else params
    dist = eqx.combine(params, static)
    return dist, losses
