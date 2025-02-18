"""SoftPlus bijection."""

from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.nn import softplus, soft_sign
from jaxtyping import Array, ArrayLike
from paramax import AbstractUnwrappable, Parameterize, unwrap
from paramax.utils import inv_softplus

from flowjax.bijections.bijection import AbstractBijection
from flowjax.utils import arraylike_to_array

class SoftPlus(AbstractBijection):
    r"""Transforms to positive domain using softplus :math:`y = \log(1 + \exp(x))`."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def transform_and_log_det(self, x, condition=None):
        return softplus(x), -softplus(-x).sum()

    def inverse_and_log_det(self, y, condition=None):
        x = jnp.log(-jnp.expm1(-y)) + y
        return x, softplus(-x).sum()
