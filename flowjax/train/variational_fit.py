"""Basic training script for fitting a flow using variational inference."""
from typing import Any, Callable

import equinox as eqx
import jax.random as jr
import optax
from jax import Array, vmap
from jax.typing import ArrayLike
from tqdm import tqdm

from flowjax.distributions import Distribution

PyTree = Any


class ElboLoss:
    """Elbo loss function, approximated using samples."""

    target: Callable[[ArrayLike], Array]
    num_samples: int

    def __init__(self, target: Callable[[ArrayLike], Array], num_samples: int):
        """
        Args:
            num_samples (int): Number of samples to use in the ELBO approximation.
            target (Callable[[ArrayLike], Array]): The target, i.e. log posterior
                density up to an additive constant / the negative of the potential
                function, evaluated for a single point.
        """
        self.target = target
        self.num_samples = num_samples

    def __call__(self, dist: Distribution, key: jr.KeyArray):
        """Computes an estimate of the negative ELBO loss."""
        samples, log_probs = dist.sample_and_log_prob(key, (self.num_samples,))
        target_density = vmap(self.target)(samples)
        losses = log_probs - target_density
        return losses.mean()


def fit_to_variational_target(
    key: jr.KeyArray,
    dist: Distribution,
    loss_fn: Callable[[Distribution, jr.KeyArray], Array],
    steps: int = 100,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    filter_spec: Callable | PyTree = eqx.is_inexact_array,
    show_progress: bool = True,
):
    """Train a distribution (e.g. a flow) by variational inference to a target
    (e.g. an unnormalized density).

    Args:
        key (jr.KeyArray): Jax PRNGKey.
        dist (Distribution): Distribution object, trainable parameters are found
            using equinox.is_inexact_array.
        loss_fn (Callable[[Distribution, jr.KeyArray], Array]): The loss function to
            optimize (e.g. the ElboLoss). The loss function should take the distribution
            to train and a key.
        steps (int, optional): The number of training steps to run. Defaults to 100.
        learning_rate (float, optional): Learning rate. Defaults to 5e-4.
        optimizer (optax.GradientTransformation | None, optional): Optax optimizer. If
            provided, this overrides the default Adam optimizer, and the learning_rate
            is ignored. Defaults to None.
        filter_spec (Callable | PyTree, optional): Equinox `filter_spec` for
            specifying trainable parameters. Either a callable `leaf -> bool`, or a
            PyTree with prefix structure matching `dist` with True/False values.
            Defaults to eqx.is_inexact_array.
        show_progress (bool, optional): Whether to show progress bar. Defaults to True.

    Returns:
        tuple: (distribution, losses).
    """
    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    @eqx.filter_value_and_grad
    def loss_val_and_grad(trainable, static, key):
        model = eqx.combine(trainable, static)
        return loss_fn(model, key)

    @eqx.filter_jit
    def step(trainable, static, key, optimizer, opt_state):
        loss_val, grads = loss_val_and_grad(trainable, static, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        trainable = eqx.apply_updates(trainable, updates)
        return trainable, opt_state, loss_val

    params, static = eqx.partition(dist, filter_spec)
    opt_state = optimizer.init(params)

    losses = []
    loop = tqdm(range(steps), disable=not show_progress)

    best_params = params
    for _ in loop:
        key, subkey = jr.split(key)
        params, opt_state, loss = step(params, static, subkey, optimizer, opt_state)
        losses.append(loss.item())
        loop.set_postfix({"loss": loss.item()})
        if loss.item() == min(losses):
            best_params = params

    return eqx.combine(best_params, static), losses
