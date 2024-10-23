"""Sigmoid bijection."""

from typing import ClassVar

import jax.numpy as jnp
from jax import nn
from jax.scipy.special import logit

from flowjax.bijections.bijection import AbstractBijection


class Sigmoid(AbstractBijection):
    r"""Sigmoid bijection :math:`y = \sigma(x) = \frac{1}{1 + \exp(-x)}`.

    Args:
        shape: The shape of the transform.
    """

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def transform_and_log_det(self, x, condition=None):
        y = nn.sigmoid(x)
        log_det = jnp.sum(nn.log_sigmoid(x) + nn.log_sigmoid(-x))
        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        x = logit(y)
        log_det = -jnp.sum(nn.log_sigmoid(x) + nn.log_sigmoid(-x))
        return x, log_det
