"""Function to fit flows to samples from a distribution."""
from typing import Any, Callable, Dict
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
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
        x (Array): Samples from target distribution.
        condition (Array | None): Conditioning variables. Defaults to None.
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

    @eqx.filter_jit
    def loss_fn(dist_trainable, dist_static, x, condition=None):
        dist = eqx.combine(dist_trainable, dist_static)
        return -dist.log_prob(x, condition).mean()

    @eqx.filter_jit
    def step(dist, optimizer, opt_state, x, condition=None):
        dist_trainable, dist_static = eqx.partition(dist, filter_spec)
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
            dist_trainable, dist_static, x, condition
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        dist = eqx.apply_updates(dist, updates)
        return dist, opt_state, loss_val

    if optimizer is None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.adam(learning_rate=learning_rate),
        )

    best_params, static = eqx.partition(dist, filter_spec)  # type: ignore
    opt_state = optimizer.init(best_params)

    key, train_val_split_key = jr.split(key)

    inputs = (x,) if condition is None else (x, condition)
    train_args, val_args = train_val_split(
        train_val_split_key, inputs, val_prop=val_prop
    )
    train_len, val_len = train_args[0].shape[0], val_args[0].shape[0]
    if batch_size > train_len:
        raise ValueError(
            f"The batch size ({batch_size}) cannot be greater than the train set size "
            f"({train_len})."
        )

    losses = {"train": [], "val": []}  # type: Dict[str, list[float]]

    loop = tqdm(range(max_epochs), disable=not show_progress)

    for _ in loop:
        key, subkey = jr.split(key)
        permutation = jr.permutation(subkey, jnp.arange(train_len))
        train_args = tuple(a[permutation] for a in train_args)

        epoch_train_loss = 0
        batch_start_idxs = range(0, train_len - batch_size + 1, batch_size)
        for i in batch_start_idxs:
            batch = tuple(a[i : i + batch_size] for a in train_args)
            dist, opt_state, loss_i = step(dist, optimizer, opt_state, *batch)
            epoch_train_loss += loss_i.item() / len(batch_start_idxs)

        epoch_val_loss = 0
        batch_start_idxs = range(0, val_len - batch_size + 1, batch_size)
        for i in batch_start_idxs:
            batch = tuple(a[i : i + batch_size] for a in val_args)

            epoch_val_loss += loss_fn(
                *eqx.partition(dist, filter_spec), *batch
            ).item() / len(batch_start_idxs)

        losses["train"].append(epoch_train_loss)
        losses["val"].append(epoch_val_loss)

        if epoch_val_loss == min(losses["val"]):
            best_params = eqx.filter(dist, filter_spec)

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

        loop.set_postfix({k: v[-1] for k, v in losses.items()})

    dist = eqx.combine(best_params, static)  # type: ignore
    return dist, losses
