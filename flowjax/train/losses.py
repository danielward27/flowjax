"Common loss functions for training normalizing flows."
from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array, vmap
from jax.lax import stop_gradient
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

from flowjax.distributions import Distribution


class Loss:
    "Base class for loss functions. All losses must implement the loss method."

    @eqx.filter_jit
    def __call__(self, static, params, *args, **kwargs):
        """Combine the partitioned model (see ``equinox.combine`` and
        ``equinox.partition``) and compute the loss."""
        model = eqx.combine(static, params)
        return self.loss(model, *args, **kwargs)

    def loss(self, *args, **kwargs):
        raise NotImplementedError("The loss method must be implemented")


class MaximumLikelihoodLoss(Loss):
    """Loss for fitting a flow with maximum likelihood (negative log likelihood). Can
    be used to learn either conditional or unconditional distributions.
    """

    @staticmethod
    def loss(dist: Distribution, x: Array, condition: Array | None = None):
        return -dist.log_prob(x, condition).mean()


class ContrastiveLoss(Loss):
    r"""Loss function for use in a sequential neural posterior estimation algorithm.
    Learns a posterior ``p(x|condition)``. Contrastive samples for each ``x`` are
    generated from other x samples in the batch.

    Note, that in a simulation based inference context, often :math:`x` is used to
    denote simulations, and :math:`\theta` for simulation parameters. However, for
    consistency with the rest of the package, we use ``x`` to represent the target
    variable (the simulator parameters), and ``condition`` for the conditioning variable
    (the simulator output/oberved data).

    References:
        - https://arxiv.org/abs/1905.07488
        - https://arxiv.org/abs/2002.03712

    """

    def __init__(self, prior: Distribution, n_contrastive: int):
        """
        Args:
            prior (Distribution): The prior distribution over x (the target variable).
            n_contrastive (int): The number of contrastive samples/atoms to use when
                computing the loss.
        """
        self.prior = prior
        self.n_contrastive = n_contrastive

    def loss(self, dist: Distribution, x: Array, condition: Array | None = None):
        contrastive = self._get_contrastive(x)
        joint_log_odds = dist.log_prob(x, condition) - self.prior.log_prob(x)
        contrastive_log_odds = dist.log_prob(
            contrastive, condition
        ) - self.prior.log_prob(contrastive)
        contrastive_log_odds = jnp.clip(contrastive_log_odds, -5)  # Clip for stability
        return -(joint_log_odds - logsumexp(contrastive_log_odds, axis=0)).mean()

    def _get_contrastive(self, theta):
        if theta.shape[0] <= self.n_contrastive:
            raise ValueError(
                f"Number of contrastive samples {self.n_contrastive} must be less than "
                f"the size of theta {theta.shape}."
            )
        # Rolling window over theta batch to create contrastive samples.
        idx = jnp.arange(len(theta))[:, None] + jnp.arange(self.n_contrastive)[None, :]
        contrastive = jnp.roll(theta[idx], -1, axis=0)  # Ensure mismatch with condition
        contrastive = jnp.swapaxes(contrastive, 0, 1)  # (contrastive, batch_size, dim)
        return contrastive


class ElboLoss(Loss):
    """The negative evidence lower bound (ELBO), approximated using samples."""

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

    def __call__(self, static: Distribution, params: Distribution, key: jr.KeyArray):
        dist = eqx.combine(static, params)

        samples = dist.sample(key, (self.num_samples,))

        def stop_grad_log_prob(params, static, samples):
            params = stop_gradient(params)
            dist = eqx.combine(params, static)
            return dist.log_prob(samples)

        log_probs = stop_grad_log_prob(
            *eqx.partition(dist, eqx.is_inexact_array), samples
        ).sum()

        target_density = vmap(self.target)(samples).sum()
        return log_probs - target_density


class RenyiElboLoss(Loss):
    def __init__(
        self, target: Callable[[ArrayLike], Array], num_samples: int, alpha: float
    ):
        """


        .. warning::
            For small alpha, e.g. alpha=0.1


        Args:
            target (Callable[[ArrayLike], Array]): _description_
            num_samples (int): _description_
            alpha (float): _description_

        Raises:
            ValueError: _description_
        """
        if alpha == 1:
            raise ValueError("Use ElboLoss for case when alpha==1.")
        self.alpha = alpha
        self.target = target
        self.num_samples = num_samples

    def loss(self, dist, key):
        # Following numpyro implementation:
        # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/infer/elbo.py
        samples, log_probs = dist.sample_and_log_prob(key, (self.num_samples,))
        target_density = vmap(self.target)(samples)
        elbo = target_density - log_probs
        scaled_elbos = (1.0 - self.alpha) * elbo
        avg_log_exp = logsumexp(scaled_elbos) - jnp.log(self.num_samples)
        weights = jnp.exp(scaled_elbos - avg_log_exp)
        renyi_elbo = avg_log_exp / (1.0 - self.alpha)
        weighted_elbo = (stop_gradient(weights) * elbo).mean(0)
        loss = -(stop_gradient(renyi_elbo - weighted_elbo) + weighted_elbo)
        return loss.sum()
