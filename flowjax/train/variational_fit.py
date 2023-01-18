from typing import Optional, Callable

from jax import random
import optax
import equinox as eqx
from tqdm import tqdm

from flowjax.utils import Array
from flowjax.distributions import Distribution

@eqx.filter_jit
def elbo_loss(dist: Distribution, target: Callable[[Array], Array], key: random.KeyArray, elbo_samples: int = 500):
    samples, approx_density = dist.sample_and_log_prob(key, sample_shape=(elbo_samples,))
    target_density = target(samples)
    losses = approx_density - target_density
    return losses.mean()

def fit_to_variational_target(
    key: random.KeyArray,
    dist: Distribution,
    target: Callable[[Array], Array],
    loss_fn: Callable[[Distribution, Callable, random.KeyArray], float] = elbo_loss,
    learning_rate: float = 5e-4,
    clip_norm: float = 0.5,
    num_epochs: int = 100,
    show_progress: bool = True,
):
    """
    Train a distribution (e.g. a flow) by variational inference.
    By default this does nothing fancy, just runs SGD steps on the ELBO loss for a fixed number of epochs.
    Args:
        key (KeyArray): Jax PRNGKey.
        dist (Distribution): Distribution object, trainable parameters are found using equinox.is_inexact_array.
        target (Callable): The variational target (this is usually the unormalized log posterior)
        loss_fcn (Callable, optional): The loss function to optimize. Variational losses are functions that take 
                                       a distribution, a target callable and a random key, and returns a scalar loss. 
                                       Defaults to elbo_loss.
        learning_rate (float, optional): Adam learning rate. Defaults to 5e-4.
        clip_norm (float, optional): Maximum gradient norm before clipping occurs. Defaults to 0.5.
        num_epochs (int, optional): The number of training steps to run. Defaults to 100.
        show_progress (bool, optional): Whether to show progress bar. Defaults to True.
    """
    @eqx.filter_jit
    def step(dist, target, key, optimizer, opt_state):
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(dist, target, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        dist = eqx.apply_updates(dist, updates)
        return dist, opt_state, loss_val

    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm), 
        optax.adam(learning_rate=learning_rate)
    )

    trainable_params, _ = eqx.partition(dist, eqx.is_inexact_array)
    opt_state = optimizer.init(trainable_params)

    losses = []

    loop = tqdm(range(num_epochs)) if show_progress is True else range(num_epochs)
    for iteration in loop:
        key, subkey = random.split(key)
        dist, opt_state, loss = step(dist, target, subkey, optimizer, opt_state)
        
        losses.append(loss.item())

        if show_progress:
            loop.set_postfix({'loss': losses[-1]})

    return dist, losses