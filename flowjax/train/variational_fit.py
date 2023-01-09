from typing import Optional, Callable

import jax.numpy as jnp
from jax import random
import optax
import equinox as eqx
from tqdm import tqdm

from flowjax.utils import Array
from flowjax.distributions import Distribution

# Variational targets accept an array of proposed samples and return an array of (usually)
# the unormalised log posterior probability of the samples.
VariationalTarget = Callable[[Array], Array]

# Variational losses are functions that take a distribution, a target 
# callable and a random key, and returns a scalar loss.
VariationalLoss = Callable[
    [Distribution, VariationalTarget, random.KeyArray], 
    float
]

@eqx.filter_jit
def elbo_loss(dist: Distribution, target: VariationalTarget, key: random.KeyArray, elbo_samples: int = 500):
    samples = dist.sample(key, n=elbo_samples)
    approx_density = dist.log_prob(samples).reshape(-1)
    target_density = target(samples).reshape(-1)
    losses = approx_density - target_density
    return losses.mean()

@eqx.filter_jit
def safe_elbo_loss(dist: Distribution, target: VariationalTarget, key: random.KeyArray, elbo_samples: int = 500):
    losses = elbo_loss(dist, target, key, elbo_samples)
    max = jnp.max(losses, where=jnp.isfinite(losses), initial=-jnp.inf)
    min = jnp.min(losses, where=jnp.isfinite(losses), initial=jnp.inf)
    losses = jnp.clip(losses, min, max)
    return losses.mean()

def variational_fit(
    key: random.KeyArray,
    dist: Distribution,
    target: VariationalTarget,
    loss_fcn: VariationalLoss = elbo_loss,
    num_epochs: int = 100,
    optimizer: Optional[optax.GradientTransformation] = None,
    show_progress: bool = True,
    recorder: Optional[Callable] = None,
):
    """
    Train a distribution (e.g. a flow) by variational inference.
    Args:
        key (KeyArray): Jax PRNGKey.
        dist (Distribution): Distribution object, trainable parameters are found using equinox.is_inexact_array.
        target (VariationalTarget): The target (usually) unormalized log posterior.
        loss_fcn (VariationalLoss, optional): Loss function. Defaults to elbo_loss.
        num_epochs (int, optional): The number of training steps to run. Defaults to 100.
        optimizer (Optional[optax.Optimizer], optional): An optax optimizer (optimizers are implemented as GradientTransformation objects). 
                                                         Defaults to an adam optimizer with learning rate 5e-4.
        show_progress (bool, optional): Whether to show progress bar. Defaults to True.
    """
    @eqx.filter_jit
    def step(dist, target, key, optimizer, opt_state):
        loss_val, grads = eqx.filter_value_and_grad(loss_fcn)(dist, target, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        dist = eqx.apply_updates(dist, updates)
        return dist, opt_state, loss_val

    # Set up a default optimizer if None is provided
    if optimizer is None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(0.5), 
            optax.adam(learning_rate=5e-4)
        )

    trainable_params, _ = eqx.partition(dist, eqx.is_inexact_array)
    opt_state = optimizer.init(trainable_params)

    losses = []
    if recorder is None:
        record = None
    else:
        record = []

    loop = tqdm(range(num_epochs)) if show_progress is True else range(num_epochs)
    for iteration in loop:
        key, subkey = random.split(key)
        dist, opt_state, loss = step(dist, target, subkey, optimizer, opt_state)
        
        if recorder is not None:
            record.append(recorder(dist))

        losses.append(loss.item())

        if show_progress:
            loop.set_postfix({'loss': losses[-1]})

    return dist, losses, record