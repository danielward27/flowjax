"""Function to fit flows to samples from a distribution."""
from typing import Any, Callable, Iterable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import Array
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike
from tqdm import tqdm

from flowjax.distributions import Distribution
from flowjax.train.train_utils import count_fruitless, get_batches, train_val_split

PyTree = Any


def fit_to_data(
    key: jr.KeyArray,
    dist: Distribution,
    x: ArrayLike,
    condition: ArrayLike | None = None,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    val_prop: float = 0.1,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    filter_spec: Callable | PyTree = eqx.is_inexact_array,
    show_progress: bool = True,
):
    """Train a distribution (e.g. a flow) to samples by maximum likelihood. Note that
    the last batch in each epoch is dropped if truncated.

    Args:
        key (KeyArray): Jax PRNGKey.
        dist (Distribution): Distribution object.
        x (ArrayLike): Samples from target distribution.
        condition (ArrayLike | None): Conditioning variables. Defaults to None.
        max_epochs (int): Maximum number of epochs. Defaults to 100.
        max_patience (int): Number of consecutive epochs with no validation
            loss improvement after which training is terminated. Defaults to 5.
        batch_size (int): Batch size. Defaults to 100.
        val_prop (float): Proportion of data to use in validation set. Defaults to 0.1.
        learning_rate (float): Adam learning rate. Defaults to 5e-4.
        optimizer (optax.GradientTransformation): Optax optimizer. If provided, this
            overrides the default Adam optimizer, and the learning_rate is ignored.
            Defaults to None.
        filter_spec (Callable | PyTree): Equinox `filter_spec` for
            specifying trainable parameters. Either a callable `leaf -> bool`, or a
            PyTree with prefix structure matching `dist` with True/False values.
            Defaults to `eqx.is_inexact_array`.
        show_progress (bool): Whether to show progress bar. Defaults to True.
    """
    x = jnp.asarray(x)

    if condition is not None:
        condition = jnp.asarray(condition)

    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    params, static = eqx.partition(dist, filter_spec)  # type: ignore
    best_params = params
    opt_state = optimizer.init(params)

    @eqx.filter_jit
    def loss_fn(dist_trainable, x, condition=None):
        dist = eqx.combine(dist_trainable, static)
        return -dist.log_prob(x, condition).mean()

    key, subkey = jr.split(key)

    inputs = (x,) if condition is None else (x, condition)
    train_args, val_args = train_val_split(subkey, inputs, val_prop=val_prop)

    losses = {"train": [], "val": []}

    loop = tqdm(range(max_epochs), disable=not show_progress)

    for _ in loop:
        # Permute arrays
        key, subkey = jr.split(key)
        train_args = [jr.permutation(subkey, a) for a in train_args]

        key, subkey = jr.split(key)
        val_args = [jr.permutation(subkey, a) for a in val_args]

        # Train epoch
        batch_losses = []
        for batch in zip(*get_batches(train_args, batch_size)):
            params, opt_state, loss_i = step(
                optimizer, opt_state, loss_fn, params, batch
            )
            batch_losses.append(loss_i)
        losses["train"].append(sum(batch_losses) / len(batch_losses))

        # Val epoch
        batch_losses = []
        for batch in zip(*get_batches(val_args, batch_size)):
            batch_losses.append(loss_fn(params, *batch))
        losses["val"].append(sum(batch_losses) / len(batch_losses))

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = params

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    dist = eqx.combine(best_params, static)  # type: ignore
    return dist, losses


def fit_to_data_sequential(
    key: jr.KeyArray,
    proposal: Distribution,
    prior: Distribution,
    theta: ArrayLike,
    x_sim: ArrayLike,
    n_contrastive: int = 5,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 50,
    val_prop: float = 0.1,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    filter_spec: Callable | PyTree = eqx.is_inexact_array,
    show_progress: bool = True,
):
    """Carry out a single round of training of a sequential neural posterior estimation
    algorithm (often referred to as SNPE-C). Learns a posterior p(theta|x_obs).

    References:
        - https://arxiv.org/abs/1905.07488
        - https://arxiv.org/abs/2002.03712

    Args:
        key (jr.KeyArray): Jax key.
        proposal (Distribution): The proposal distribution to train.
        prior (Distribution): The prior distribution.
        theta (ArrayLike): A batch of simulation parameters.
        x_sim (ArrayLike): A batch of simulation outputs, corresponding to theta.
        n_contrastive (int, optional): The number of contrasting theta values used in
            the loss computation. Defaults to 5.
        max_epochs (int, optional): The maximum number of epochs. Defaults to 100.
        max_patience (int, optional): Number of consecutive epochs with no validation
            loss improvement after which training is terminated. Defaults to 5.
        batch_size (int, optional): Batch size. Defaults to 50.
        val_prop (float, optional): Proportion of data to use for validation.
            Defaults to 0.1.
        learning_rate (float, optional): Adam learning rate. Defaults to 5e-4.
        optimizer (optax.GradientTransformation): Optax optimizer. If provided, this
            overrides the default Adam optimizer, and the learning_rate is ignored.
            Defaults to None.
        filter_spec (Callable | PyTree): Equinox `filter_spec` for
            specifying trainable parameters. Either a callable `leaf -> bool`, or a
            PyTree with prefix structure matching `dist` with True/False values.
            Defaults to `eqx.is_inexact_array`.
        show_progress (bool): Whether to show progress bar. Defaults to True.
    """
    theta, x_sim = jnp.asarray(theta), jnp.asarray(x_sim)

    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    params, static = eqx.partition(proposal, filter_spec)
    opt_state = optimizer.init(params)

    # Sample contrastive parameters
    key, subkey = jr.split(key)
    contrastive = jr.choice(subkey, theta, (n_contrastive, theta.shape[0]))

    # Train val split
    key, subkey = jr.split(key)
    train_args, val_args = train_val_split(
        subkey, (theta, x_sim, contrastive), val_prop=val_prop, axis=[0, 0, 1]
    )

    @eqx.filter_jit
    def loss_fn(params, theta, x_sim, contrastive):
        proposal = eqx.combine(params, static)
        sim_log_odds = proposal.log_prob(theta, x_sim) - prior.log_prob(theta)
        contrastive_log_odds = proposal.log_prob(contrastive, x_sim) - prior.log_prob(
            contrastive
        )
        contrastive_log_odds = jnp.clip(contrastive_log_odds, -5)  # Clip for stability
        return -(sim_log_odds - logsumexp(contrastive_log_odds, axis=0)).mean()

    loop = tqdm(range(max_epochs), disable=not show_progress)

    best_params = params
    losses = {"train": [], "val": []}
    for _ in loop:
        # Permute arrays
        key, subkey = jr.split(key)
        train_args = [jr.permutation(subkey, a) for a in train_args]

        key, subkey = jr.split(key)
        val_args = [jr.permutation(subkey, a) for a in val_args]

        # Permute contrasting samples independently of other data
        for axis in (0, 1):
            for args in [train_args, val_args]:
                key, subkey = jr.split(key)
                args[-1] = eqx.filter_jit(jr.permutation)(subkey, args[-1], axis)

        # Train epoch
        batch_losses = []
        for batch in zip(*get_batches(train_args, batch_size, axis=[0, 0, 1])):
            params, opt_state, loss_i = step(
                optimizer, opt_state, loss_fn, params, batch
            )
            batch_losses.append(loss_i)
        losses["train"].append(sum(batch_losses) / len(batch_losses))

        # Val epoch
        batch_losses = []
        for batch in zip(*get_batches(val_args, batch_size, axis=[0, 0, 1])):
            batch_losses.append(loss_fn(params, *batch))
        losses["val"].append(sum(batch_losses) / len(batch_losses))

        if losses["val"][-1] == min(losses["val"]):
            best_params = params

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

        loop.set_postfix({k: v[-1] for k, v in losses.items()})

    proposal = eqx.combine(best_params, static)  # type: ignore
    return proposal, losses


@eqx.filter_jit
def step(
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    loss_fn: Callable,
    params: PyTree,
    batch: Iterable[Array],
):
    """Carry out a training step on a batch of data."""
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params, *batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_val
