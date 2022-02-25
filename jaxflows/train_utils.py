from jaxflows.flow import Flow
from jax import random
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm

def train_flow(
    flow : Flow,
    key : random.PRNGKey,
    x : jnp.ndarray,
    max_epochs : int = 50,
    max_patience : int = 5,
    learning_rate : float = 5e-4,
    batch_size : int = 256,
    val_prop : float = 0.1):

    def loss(flow, x):
        return -flow.log_prob(x).mean()

    @eqx.filter_jit
    def step(flow, x_batch, optimizer, opt_state):
        loss_val, grads = eqx.filter_value_and_grad(loss)(flow, x_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        flow = eqx.apply_updates(flow, updates)
        return flow, opt_state, loss_val

    train_x, val_x = train_val_split(key, x, val_prop=val_prop)

    optimizer = optax.adam(learning_rate=learning_rate)

    best_params, static = eqx.partition(flow, eqx.is_array)
    opt_state = optimizer.init(best_params)
    losses = []

    pbar = tqdm(range(max_epochs))

    losses = {"train": [], "val": []}

    for epoch in pbar:
        key, subkey = random.split(key)
        train_x = random.permutation(key, train_x)

        batches = range(0, train_x.shape[0]-batch_size, batch_size)
        epoch_train_loss = 0
        for i in batches:
            train_x_batch = train_x[i : i+batch_size]
            flow, opt_state, loss_val = step(flow, train_x_batch, optimizer, opt_state)        
            epoch_train_loss += loss_val.item()

        val_loss = loss(flow, val_x).item()
        losses["train"].append(epoch_train_loss / len(batches))
        losses["val"].append(val_loss)
        
        if val_loss == min(losses["val"]):
            best_params, _ = eqx.partition(flow, eqx.is_array)

        elif count_fruitless(losses["val"]) > max_patience:
            print("Max patience reached.")
            break

        pbar.set_postfix({k: v[-1] for k,v in losses.items()})

    flow = eqx.combine(best_params, static)
    return flow, losses


def train_val_split(
    key: random.PRNGKey, *args, val_prop: float = 0.1):
    "Returns (train_x, val_x, train_y, val_y, ...)"
    assert 0 <= val_prop <= 1
    n = args[0].shape[0]
    n_val = round(val_prop*n)
    shuffle = random.permutation(key, jnp.arange(n))
    idxs = (shuffle[:-n_val], shuffle[n_val:])
    train_val = tuple(array[idx] for array in args for idx in idxs)
    return train_val

def count_fruitless(losses: list):
    """Given a list of losses from each epoch, count the number of epochs since
    the minimum loss"""
    min_idx = jnp.array(losses).argmin().item()
    return len(losses) - min_idx - 1