"""Common loss functions for training normalizing flows.

The loss functions are callables, with the first two arguments being the partitioned
distribution (see ``equinox.partition``).
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import Array, vmap
from jax.lax import stop_gradient
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

from flowjax.distributions import AbstractDistribution
from flowjax.wrappers import unwrap


class MaximumLikelihoodLoss:
    """Loss for fitting a flow with maximum likelihood (negative log likelihood).

    This loss can be used to learn either conditional or unconditional distributions.
    """

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractDistribution,
        static: AbstractDistribution,
        x: Array,
        condition: Array | None = None,
    ):
        """Compute the loss."""
        dist = unwrap(eqx.combine(static, params))
        return -dist.log_prob(x, condition).mean()


class ContrastiveLoss:
    r"""Loss function for use in a sequential neural posterior estimation algorithm.

    Learns a posterior ``p(x|condition)``. Contrastive samples for each ``x`` are
    generated from other x samples in the batch.

    Note, that in a simulation based inference context, often :math:`x` is used to
    denote simulations, and :math:`\theta` for simulation parameters. However, for
    consistency with the rest of the package, we use ``x`` to represent the target
    variable (the simulator parameters), and ``condition`` for the conditioning variable
    (the simulator output/oberved data).

    Args:
        prior: The prior distribution over x (the target
            variable).
        n_contrastive: The number of contrastive samples/atoms to use when
            computing the loss.

    References:
        - https://arxiv.org/abs/1905.07488
        - https://arxiv.org/abs/2002.03712
    """

    def __init__(self, prior: AbstractDistribution, n_contrastive: int):
        self.prior = prior
        self.n_contrastive = n_contrastive

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractDistribution,
        static: AbstractDistribution,
        x: Array,
        condition: Array | None = None,
    ):
        """Compute the loss."""
        dist = unwrap(eqx.combine(params, static))
        contrastive = self._get_contrastive(x)
        joint_log_odds = dist.log_prob(x, condition) - self.prior.log_prob(x)
        contrastive_log_odds = dist.log_prob(
            contrastive,
            condition,
        ) - self.prior.log_prob(contrastive)
        contrastive_log_odds = jnp.clip(contrastive_log_odds, -5)  # Clip for stability
        return -(joint_log_odds - logsumexp(contrastive_log_odds, axis=0)).mean()

    def _get_contrastive(self, theta):
        if theta.shape[0] <= self.n_contrastive:
            raise ValueError(
                f"Number of contrastive samples {self.n_contrastive} must be less than "
                f"the size of theta {theta.shape}.",
            )
        # Rolling window over theta batch to create contrastive samples.
        idx = jnp.arange(len(theta))[:, None] + jnp.arange(self.n_contrastive)[None, :]
        contrastive = jnp.roll(theta[idx], -1, axis=0)  # Ensure mismatch with condition
        return jnp.swapaxes(contrastive, 0, 1)  # (contrastive, batch_size, dim)


class ElboLoss:
    """The negative evidence lower bound (ELBO), approximated using samples.

    Args:
        num_samples: Number of samples to use in the ELBO approximation.
        target: The target, i.e. log posterior density up to an additive constant / the
            negative of the potential function, evaluated for a single point.
        stick_the_landing: Whether to use the (often) lower variance ELBO gradient
            estimator introduced in https://arxiv.org/pdf/1703.09194.pdf. Note for flows
            this requires evaluating the flow in both directions (running the forward
            and inverse transformation). For some flow architectures, this may be
            computationally expensive due to assymetrical computational complexity
            between the forward and inverse transformation. Defaults to False.
    """

    target: Callable[[ArrayLike], Array]
    num_samples: int
    stick_the_landing: bool

    def __init__(
        self,
        target: Callable[[ArrayLike], Array],
        num_samples: int,
        *,
        stick_the_landing: bool = False,
    ):
        self.target = target
        self.num_samples = num_samples
        self.stick_the_landing = stick_the_landing

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractDistribution,
        static: AbstractDistribution,
        key: Array,
    ):
        """Compute the ELBO loss.

        Args:
            params: The trainable parameters of the model.
            static: The static components of the model.
            key: Jax random seed.
        """
        dist = eqx.combine(params, static)

        if self.stick_the_landing:
            # Requries both forward and inverse pass
            samples = dist.sample(key, (self.num_samples,))
            dist = eqx.combine(stop_gradient(params), static)
            log_probs = dist.log_prob(samples)

        else:
            # Requires only forward pass through the flow.
            samples, log_probs = dist.sample_and_log_prob(key, (self.num_samples,))

        target_density = vmap(self.target)(samples)
        return (log_probs - target_density).mean()
