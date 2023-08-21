"""Function to fit flows to samples from a distribution."""
from typing import Any, Callable, Dict, Iterable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import Array
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike
from tqdm import tqdm

from flowjax.distributions import Distribution
from flowjax.train.train_utils import count_fruitless, train_val_split

PyTree = Any


def fit_to_data(
    key: jr.KeyArray,
    dist: Distribution,
    x: ArrayLike,
    condition: ArrayLike | None = None,
    max_epochs: int = 50,
    max_patience: int = 5,
    batch_size: int = 256,
    val_prop: float = 0.1,
    learning_rate: float = 5e-4,
    clip_norm: float = 0.5,
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
        max_epochs (int): Maximum number of epochs. Defaults to 50.
        max_patience (int): Number of consecutive epochs with no validation
            loss improvement after which training is terminated. Defaults to 5.
        batch_size (int): Batch size. Defaults to 256.
        val_prop (float): Proportion of data to use in validation set. Defaults to 0.1.
        learning_rate (float): Adam learning rate. Defaults to 5e-4.
        clip_norm (float): Maximum gradient norm before clipping occurs. Defaults to 0.5.
        optimizer (optax.GradientTransformation): Optax optimizer. If provided, this
            overrides the default Adam optimizer, and the learning_rate and clip_norm
            arguments are ignored. Defaults to None.
        filter_spec (Callable | PyTree): Equinox `filter_spec` for
            specifying trainable parameters. Either a callable `leaf -> bool`, or a
            PyTree with prefix structure matching `dist` with True/False values.
            Defaults to `eqx.is_inexact_array`.
        show_progress (bool): Whether to show progress bar. Defaults to True.
    """
    x = jnp.asarray(x)

    if condition is not None:
        condition = jnp.asarray(condition)

    params, static = eqx.partition(dist, filter_spec)  # type: ignore
    if optimizer is None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.adam(learning_rate=learning_rate),
        )

    best_params = params
    opt_state = optimizer.init(params)

    @eqx.filter_jit
    def loss_fn(dist_trainable, x, condition=None):
        dist = eqx.combine(dist_trainable, static)
        return -dist.log_prob(x, condition).mean()

    key, subkey = jr.split(key)

    inputs = (x,) if condition is None else (x, condition)
    train_args, val_args = train_val_split(subkey, inputs, val_prop=val_prop)

    losses = {"train": [], "val": []}  # type: Dict[str, list[float]]

    loop = tqdm(range(max_epochs), disable=not show_progress)

    for _ in loop:
        key, subkey = jr.split(key)
        train_args = [jr.permutation(subkey, a) for a in train_args]
        params, opt_state, epoch_train_loss = train_epoch(
            optimizer, opt_state, loss_fn, batch_size, params, train_args
        )

        val_args = [jr.permutation(subkey, a) for a in val_args]
        epoch_val_loss = validation_epoch(loss_fn, batch_size, params, val_args)

        losses["train"].append(epoch_train_loss)
        losses["val"].append(epoch_val_loss)

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if epoch_val_loss == min(losses["val"]):
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
    x_obs: ArrayLike,
    is_first_round: bool,
    n_contrastive: int = 10,
    max_epochs: int = 50,
    max_patience: int = 5,
    batch_size: int = 50,
    val_prop: float = 0.1,
    learning_rate: float = 5e-4,
    clip_norm: float = 0.5,
    optimizer: optax.GradientTransformation | None = None,
    filter_spec: Callable | PyTree = eqx.is_inexact_array,
    show_progress: bool = True,
):
    """Carry out a single round of training of a sequential neural posterior estimation
    algorithm (often referred to as SNPE-C). Learns a posterior p(theta|x_obs).

    References:
        - https://arxiv.org/abs/1905.07488
        - https://arxiv.org/abs/2002.03712
    """
    theta, x_sim, x_obs = jnp.asarray(theta), jnp.asarray(x_sim), jnp.asarray(x_obs)

    if optimizer is None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.adam(learning_rate=learning_rate),
        )
    params, static = eqx.partition(proposal, filter_spec)
    opt_state = optimizer.init(params)

    # TODO this will be memory intensive and should be batched
    key, subkey = jr.split(key)
    if is_first_round:
        contrastive = prior.sample(subkey, (theta.shape[0], n_contrastive))
    else:
        contrastive = proposal.sample(subkey, (theta.shape[0], n_contrastive), x_obs)

    # Train val split
    key, subkey = jr.split(key)
    train_args, val_args = train_val_split(
        subkey, (theta, x_sim, contrastive), val_prop=val_prop
    )

    @eqx.filter_jit
    def loss_fn(params, theta, x_sim, contrastive):
        proposal = eqx.combine(params, static)
        sim_log_odds = proposal.log_prob(theta, x_sim) - prior.log_prob(theta)

        contrastive = jnp.swapaxes(contrastive, 0, 1)  # (contrastive, batch, theta_dim)
        contrast_log_odds = proposal.log_prob(contrastive, x_sim) - prior.log_prob(
            contrastive
        )
        contrast_log_odds = jnp.clip(contrast_log_odds, -5)  # Clip for stability
        return -(sim_log_odds - logsumexp(contrast_log_odds, axis=0)).mean()

    loop = tqdm(range(max_epochs), disable=not show_progress)

    key, subkey = jr.split(key)
    best_params = params
    losses = {"train": [], "val": []}  # type: Dict[str, list[float]]
    for _ in loop:
        # Permute arrays
        key, subkey = jr.split(key)
        train_args = [jr.permutation(subkey, a) for a in train_args]

        key, subkey = jr.split(key)
        val_args = [jr.permutation(subkey, a) for a in val_args]

        params, opt_state, epoch_train_loss = train_epoch(
            optimizer=optimizer,
            opt_state=opt_state,
            loss_fn=loss_fn,
            batch_size=batch_size,
            params=params,
            batchable=train_args,
        )

        epoch_val_loss = validation_epoch(loss_fn, batch_size, params, val_args)

        losses["train"].append(epoch_train_loss)
        losses["val"].append(epoch_val_loss)

        if epoch_val_loss == min(losses["val"]):
            best_params = params

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

        loop.set_postfix({k: v[-1] for k, v in losses.items()})

    proposal = eqx.combine(best_params, static)  # type: ignore
    return proposal, losses


def train_epoch(
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    loss_fn: Callable,
    batch_size: int,
    params: PyTree,
    batchable: Iterable[Array],
):
    """Carry out a single epoch of training, returning the parameters optimizer state
    and the loss value. We assume args are arrays that can be batched along axis 0.

    Args:
        optimizer (optax.GradientTransformation): The optimizer to use.
        opt_state: The optimizer state.
        loss_fn (Callable): The loss function taking (params, batchable, non_batchable).
        batch_size (int): The batch size.
        params (PyTree): The parameters to update.
        batchable: Batchable arguments unpacked into the loss after params.
    """
    train_len = batchable[0].shape[0]
    batch_size = min(train_len, batch_size)
    batch_start_idxs = range(0, train_len - batch_size + 1, batch_size)
    epoch_loss = 0
    for i in batch_start_idxs:
        batch = [a[i : i + batch_size] for a in batchable]
        params, opt_state, loss_i = step(optimizer, opt_state, loss_fn, params, batch)
        epoch_loss += loss_i / len(batch_start_idxs)
    return params, opt_state, epoch_loss


def validation_epoch(
    loss_fn: Callable, batch_size: int, params: PyTree, batchable: Iterable[Array]
):
    """Compute the loss for the validation set."""
    val_len = batchable[0].shape[0]
    batch_size = min(batch_size, val_len)
    batch_start_idxs = range(0, val_len - batch_size + 1, batch_size)

    epoch_loss = 0
    for i in batch_start_idxs:
        batch = [a[i : i + batch_size] for a in batchable]
        epoch_loss += loss_fn(params, *batch).item() / len(batch_start_idxs)

    return epoch_loss


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
