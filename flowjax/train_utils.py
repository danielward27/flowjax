from flowjax.flows import Flow
from jax import random
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm
from typing import Optional


def train_flow(
    key: random.PRNGKey,
    flow: Flow,
    x: jnp.ndarray,
    condition: Optional[jnp.ndarray] = None,
    max_epochs: int = 50,
    max_patience: int = 5,
    learning_rate: float = 5e-4,
    batch_size: int = 256,
    val_prop: float = 0.1,
    clip_norm: float = 0.5,
    show_progress: bool = True,
):
    """Train flow with Adam optimizer.

    Args:
        key (random.PRNGKey): Jax key.
        flow (Flow): Flow to train.
        x (jnp.ndarray): Samples from the target distribution (each row being a sample).
        condition (Optional[jnp.ndarray], optional): Conditioning variables corresponding to x if learning a conditional distribution. Defaults to None.
        max_epochs (int, optional): Maximum number of epochs. Defaults to 50.
        max_patience (int, optional): Number of consecutive epochs with no validation loss improvement after which training is terminated. Defaults to 5.
        learning_rate (float, optional): Adam learning rate. Defaults to 5e-4.
        batch_size (int, optional): Batch size. Defaults to 256.
        val_prop (float, optional): Proportion of data to use for validation. Defaults to 0.1.
        clip_norm (float, optional): Maximum gradient norm before clipping.
        show_progress (bool, optional): Whether to show progress bar. Defaults to True.
    """

    def loss(flow, x, condition=None):
        return -flow.log_prob(x, condition).mean()

    @eqx.filter_jit
    def step(flow, optimizer, opt_state, x, condition=None):
        loss_val, grads = eqx.filter_value_and_grad(loss)(flow, x, condition)
        updates, opt_state = optimizer.update(grads, opt_state)
        flow = eqx.apply_updates(flow, updates)
        return flow, opt_state, loss_val

    key, subkey = random.split(key)

    inputs = (x,) if condition is None else (x, condition)
    train_args, val_args = train_val_split(subkey, inputs, val_prop=val_prop)

    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate=learning_rate)
    )

    best_params, static = eqx.partition(flow, eqx.is_inexact_array)
    opt_state = optimizer.init(best_params)

    losses = {"train": [], "val": []}

    loop = tqdm(range(max_epochs)) if show_progress is True else range(max_epochs)
    for epoch in loop:
        key, subkey = random.split(key)
        train_args = random_permutation_multiple(subkey, train_args)

        epoch_train_loss = 0
        batches = range(0, train_args[0].shape[0] - batch_size, batch_size)
        for i in batches:
            batch = tuple(a[i : i + batch_size] for a in train_args)
            flow, opt_state, loss_i = step(flow, optimizer, opt_state, *batch)
            epoch_train_loss += loss_i.item() / len(batches)

        epoch_val_loss = 0
        batches = range(0, val_args[0].shape[0] - batch_size, batch_size)
        for i in batches:
            batch = tuple(a[i : i + batch_size] for a in val_args)
            epoch_val_loss += loss(flow, *batch).item() / len(batches)

        losses["train"].append(epoch_train_loss)
        losses["val"].append(epoch_val_loss)

        if epoch_val_loss == min(losses["val"]):
            best_params = eqx.filter(flow, eqx.is_inexact_array)

        elif count_fruitless(losses["val"]) > max_patience:
            print("Max patience reached.")
            break

        if show_progress:
            loop.set_postfix({k: v[-1] for k, v in losses.items()})

    flow = eqx.combine(best_params, static)
    return flow, losses


def train_val_split(key: random.PRNGKey, arrays, val_prop: float = 0.1):
    "Returns ((train_x, train_y), (val_x, val_y), ...)). Split on axis 0."
    assert 0 <= val_prop <= 1
    key, subkey = random.split(key)
    arrays = random_permutation_multiple(subkey, arrays)
    n_val = round(val_prop * arrays[0].shape[0])
    train = tuple(a[:-n_val] for a in arrays)
    val = tuple(a[-n_val:] for a in arrays)
    return train, val


def random_permutation_multiple(key, arrays):
    "Randomly permute multiple arrays on axis 0 (consistent between arrays)."
    n = arrays[0].shape[0]
    shuffle = random.permutation(key, jnp.arange(n))
    arrays = tuple(a[shuffle] for a in arrays)
    return arrays


def count_fruitless(losses: list):
    """Given a list of losses from each epoch, count the number of epochs since
    the minimum loss"""
    min_idx = jnp.array(losses).argmin().item()
    return len(losses) - min_idx - 1
