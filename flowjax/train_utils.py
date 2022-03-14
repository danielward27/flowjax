from flowjax.flows import Flow
from jax import random
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm


def train_flow(
    key: random.PRNGKey,
    flow: Flow,
    x: jnp.ndarray,
    condition: jnp.ndarray = None,
    max_epochs: int = 50,
    max_patience: int = 5,
    learning_rate: float = 5e-4,
    batch_size: int = 256,
    val_prop: float = 0.1,
):
    if condition is None:
        condition = jnp.empty((x.shape[0], 0))  # Note zero dim col

    def loss(flow, x, condition):
        return -flow.log_prob(x, condition).mean()

    @eqx.filter_jit
    def step(flow, x, condition, optimizer, opt_state):
        loss_val, grads = eqx.filter_value_and_grad(loss)(flow, x, condition)
        updates, opt_state = optimizer.update(grads, opt_state)
        flow = eqx.apply_updates(flow, updates)
        return flow, opt_state, loss_val

    key, subkey = random.split(key)
    train_args, val_args = train_val_split(subkey, (x, condition), val_prop=val_prop)

    optimizer = optax.adam(learning_rate=learning_rate)
    best_params, static = eqx.partition(flow, eqx.is_array)

    opt_state = optimizer.init(best_params)
    losses = []

    pbar = tqdm(range(max_epochs))

    losses = {"train": [], "val": []}

    for epoch in pbar:
        key, subkey = random.split(key)
        train_args = random_permutation_multiple(subkey, train_args)
        batches = range(0, train_args[0].shape[0] - batch_size, batch_size)

        epoch_train_loss = 0
        for i in batches:
            x_batch, cond_batch = (
                train_args[0][i : i + batch_size],
                train_args[1][i : i + batch_size],
            )
            flow, opt_state, loss_val = step(
                flow, x_batch, cond_batch, optimizer, opt_state
            )
            epoch_train_loss += loss_val.item()

        val_loss = loss(flow, *val_args).item()
        losses["train"].append(epoch_train_loss / len(batches))
        losses["val"].append(val_loss)

        if val_loss == min(losses["val"]):
            best_params, _ = eqx.partition(flow, eqx.is_array)

        elif count_fruitless(losses["val"]) > max_patience:
            print("Max patience reached.")
            break

        pbar.set_postfix({k: v[-1] for k, v in losses.items()})

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
