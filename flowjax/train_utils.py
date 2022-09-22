from flowjax.distributions import Distribution
from jax import random
from jax.random import KeyArray
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple
from flowjax.utils import Array


def train_flow(
    key: KeyArray,
    dist: Distribution,
    x: Array,
    condition: Optional[Array] = None,
    max_epochs: int = 50,
    max_patience: int = 5,
    learning_rate: float = 5e-4,
    batch_size: int = 256,
    val_prop: float = 0.1,
    clip_norm: float = 0.5,
    show_progress: bool = True,
):
    """Train a distribution (e.g. a flow) by maximum likelihood with Adam optimizer.

    Args:
        key (KeyArray): Jax PRNGKey.
        dist (Distribution): Distribution object, trainable parameters are found using equinox.is_inexact_array.
        x (Array): Samples from target distribution.
        condition (Optional[Array], optional): Conditioning variables. Defaults to None.
        max_epochs (int, optional): Maximum number of epochs. Defaults to 50.
        max_patience (int, optional): Number of consecutive epochs with no validation loss improvement after which training is terminated. Defaults to 5.
        learning_rate (float, optional): Adam learning rate. Defaults to 5e-4.
        batch_size (int, optional): Batch size. Defaults to 256.
        val_prop (float, optional): Proportion of data to use in validation set. Defaults to 0.1.
        clip_norm (float, optional): Maximum gradient norm before clipping occurs. Defaults to 0.5.
        show_progress (bool, optional): Whether to show progress bar. Defaults to True.
    """

    def loss(dist, x, condition=None):
        return -dist.log_prob(x, condition).mean()

    @eqx.filter_jit
    def step(dist, optimizer, opt_state, x, condition=None):
        loss_val, grads = eqx.filter_value_and_grad(loss)(dist, x, condition)
        updates, opt_state = optimizer.update(grads, opt_state)
        dist = eqx.apply_updates(dist, updates)
        return dist, opt_state, loss_val

    key, subkey = random.split(key)

    inputs = (x,) if condition is None else (x, condition)
    train_args, val_args = train_val_split(subkey, inputs, val_prop=val_prop)

    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate=learning_rate)
    )

    best_params, static = eqx.partition(dist, eqx.is_inexact_array)
    opt_state = optimizer.init(best_params)

    losses = {"train": [], "val": []}  # type: Dict[str, List[float]]

    loop = tqdm(range(max_epochs)) if show_progress is True else range(max_epochs)
    for epoch in loop:
        key, subkey = random.split(key)
        train_args = random_permutation_multiple(subkey, train_args)

        epoch_train_loss = 0
        batches = range(0, train_args[0].shape[0] - batch_size, batch_size)
        for i in batches:
            batch = tuple(a[i : i + batch_size] for a in train_args)
            dist, opt_state, loss_i = step(dist, optimizer, opt_state, *batch)
            epoch_train_loss += loss_i.item() / len(batches)

        epoch_val_loss = 0
        batches = range(0, val_args[0].shape[0] - batch_size, batch_size)
        for i in batches:
            batch = tuple(a[i : i + batch_size] for a in val_args)
            epoch_val_loss += loss(dist, *batch).item() / len(batches)

        losses["train"].append(epoch_train_loss)
        losses["val"].append(epoch_val_loss)

        if epoch_val_loss == min(losses["val"]):
            best_params = eqx.filter(dist, eqx.is_inexact_array)

        elif count_fruitless(losses["val"]) > max_patience:
            print("Max patience reached.")
            break

        if show_progress:
            loop.set_postfix({k: v[-1] for k, v in losses.items()})

    dist = eqx.combine(best_params, static)
    return dist, losses


def train_val_split(key: KeyArray, arrays: List[Array], val_prop: float = 0.1):
    """Train validation split along axis 0.

    Args:
        key (KeyArray): Jax PRNGKey
        arrays List[Array]: List of arrays.
        val_prop (float, optional): Proportion of data to use for validation. Defaults to 0.1.

    Returns:
        Tuple[Tuple]: (train_arrays, validation_arrays)
    """
    if not (0 <= val_prop <= 1):
        raise ValueError("val_prop should be between 0 and 1.")
    key, subkey = random.split(key)
    arrays = random_permutation_multiple(subkey, arrays)
    n_val = round(val_prop * arrays[0].shape[0])
    train = tuple(a[:-n_val] for a in arrays)
    val = tuple(a[-n_val:] for a in arrays)
    return train, val


def random_permutation_multiple(key: KeyArray, arrays: List[Array]) -> Tuple[Array]:
    """Randomly permute multiple arrays on axis 0 (consistent between arrays)

    Args:
        key (KeyArray): Jax PRNGKey
        arrays (List[Array]): List of arrays.

    Returns:
        List[Array]: List of permuted arrays.
    """
    n = arrays[0].shape[0]
    shuffle = random.permutation(key, jnp.arange(n))
    arrays = tuple(a[shuffle] for a in arrays)
    return arrays


def count_fruitless(losses: List[float]) -> int:
    """Given a list of losses from each epoch, count the number of epochs since
    the minimum loss.

    Args:
        losses (List[float]): List of losses.

    """
    min_idx = jnp.array(losses).argmin().item()
    return len(losses) - min_idx - 1
