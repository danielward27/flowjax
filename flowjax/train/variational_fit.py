"""Basic training script for fitting a flow using variational inference."""
from typing import Any, Callable

import equinox as eqx
import jax.random as jr
import optax
from jax import Array
from tqdm import tqdm

from flowjax.distributions import Distribution

PyTree = Any


@eqx.filter_jit
def elbo_loss(
    dist: Distribution,
    target: Callable[[Array], Array],
    key: jr.KeyArray,
    num_samples: int,
) -> Array:
    """The evidence lower bound loss function."""
    samples, approx_density = dist.sample_and_log_prob(key, (num_samples,))
    target_density = target(samples)
    losses = approx_density - target_density
    return losses.mean()


def fit_to_variational_target(
    key: jr.KeyArray,
    dist: Distribution,
    target: Callable[[Array], Array],
    loss_fn: Callable[[Distribution, Callable, jr.KeyArray, int], Array] = elbo_loss,
    steps: int = 100,
    samples_per_step: int = 500,
    learning_rate: float = 5e-4,
    clip_norm: float = 0.5,
    optimizer: optax.GradientTransformation | None = None,
    filter_spec: Callable | PyTree = eqx.is_inexact_array,
    show_progress: bool = True,
):
    """
    Train a distribution (e.g. a flow) by variational inference to a target
    (e.g. an unnormalized density).

    Args:
        key (KeyArray): Jax PRNGKey.
        dist (Distribution): Distribution object, trainable parameters are found
            using equinox.is_inexact_array.
        target (Callable): The variational target (this is usually the unormalized log
            posterior)
        loss_fn (Callable): The loss function to optimize. The loss function
            should take a random key, the distribution, a callable that maps samples
            from the distribution to a scalar loss, and a number of samples to use.
            Defaults to elbo_loss.
        steps (int): The number of training steps to run. Defaults to 100.
        samples_per_step (int): number of samples to use at each step.
            Defaults to 500.
        learning_rate (float): Adam learning rate. Defaults to 5e-4.
        clip_norm (float): Maximum gradient norm before clipping occurs.
            Defaults to 0.5.
        optimizer (optax.GradientTransformation | None): Optax optimizer. If provided,
            this overrides the default Adam optimizer, and the learning_rate and
            clip_norm arguments are ignored. Defaults to None.
        filter_spec (Callable | PyTree): Equinox `filter_spec` for
            specifying trainable parameters. Either a callable `leaf -> bool`, or a
            PyTree with prefix structure matching `dist` with True/False values.
            Defaults to `eqx.is_inexact_array`.
        show_progress (bool): Whether to show progress bar. Defaults to True.
    """
    if optimizer is None:
        optimizer = optimizer = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.adam(learning_rate=learning_rate),
        )

    @eqx.filter_jit
    def step(dist, target, key, optimizer, opt_state):
        @eqx.filter_value_and_grad
        def loss_val_and_grad(
            dist_trainable, dist_static, target, key, samples_per_step
        ):
            dist = eqx.combine(dist_trainable, dist_static)
            return loss_fn(dist, target, key, samples_per_step)

        dist_trainable, dist_static = eqx.partition(dist, filter_spec)
        loss_val, grads = loss_val_and_grad(
            dist_trainable, dist_static, target, key, samples_per_step
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        dist = eqx.apply_updates(dist, updates)
        return dist, opt_state, loss_val

    dist_trainable, _ = eqx.partition(dist, filter_spec)  # type: ignore
    opt_state = optimizer.init(dist_trainable)

    losses = []

    loop = tqdm(range(steps), disable=not show_progress)
    for _ in loop:
        key, subkey = jr.split(key)
        dist, opt_state, loss = step(dist, target, subkey, optimizer, opt_state)  # type: ignore

        losses.append(loss.item())
        loop.set_postfix({"loss": losses[-1]})

    return dist, losses
