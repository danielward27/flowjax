"""Planar transform.

A layer in a planar flow introduced in https://arxiv.org/pdf/1505.05770.pdf.
"""

from collections.abc import Callable
from functools import partial
from typing import ClassVar, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import nn
from jax.numpy.linalg import norm
from jaxtyping import Array, Float, PRNGKeyArray

from flowjax.bijections.bijection import AbstractBijection


class Planar(AbstractBijection):
    r"""Planar bijection as used by https://arxiv.org/pdf/1505.05770.pdf.

    Uses the transformation

    .. math::

        \boldsymbol{y}=\boldsymbol{x} +
            \boldsymbol{u} \cdot \text{tanh}(\boldsymbol{w}^T \boldsymbol{x} + b)

    where :math:`\boldsymbol{u} \in \mathbb{R}^D, \ \boldsymbol{w} \in \mathbb{R}^D`
    and :math:`b \in \mathbb{R}`. In the unconditional case, the (unbounded) parameters
    are learned directly. In the unconditional case they are parameterised by an MLP.

    Args:
        key: Jax random key.
        dim: Dimension of the bijection.
        cond_dim: Dimension of extra conditioning variables. Defaults to None.
        negative_slope: A positive float. If provided, then a leaky relu activation
            (with the corresponding negative slope) is used instead of tanh. This also
            provides the advantage that the bijection can be inverted analytically.
        **mlp_kwargs: Keyword arguments (excluding in_size and out_size) passed to
            the MLP (``equinox.nn.MLP``). Ignored when ``cond_dim`` is None.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    conditioner: Callable | None
    params: Array | None
    negative_slope: float | None

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        dim: int,
        cond_dim: int | None = None,
        negative_slope: float | None = None,
        **mlp_kwargs,
    ):
        self.shape = (dim,)

        if cond_dim is None:
            self.params = 0.01 * jr.normal(key, (2 * dim + 1,))
            self.conditioner = None
            self.cond_shape = None
        else:
            self.params = None
            self.conditioner = eqx.nn.MLP(cond_dim, 2 * dim + 1, **mlp_kwargs, key=key)
            self.cond_shape = (cond_dim,)

        self.negative_slope = negative_slope

    def transform_and_log_det(self, x, condition=None):
        return self.get_planar(condition).transform_and_log_det(x)

    def inverse_and_log_det(self, y, condition=None):
        return self.get_planar(condition).inverse_and_log_det(y)

    def get_planar(self, condition=None):
        """Get the planar bijection with the conditioning applied if conditional."""
        if self.cond_shape is not None:
            params = self.conditioner(condition)
        else:
            params = self.params
        dim = self.shape[0]
        assert params is not None
        w, u, bias = params[:dim], params[dim : 2 * dim], params[-1]
        return _UnconditionalPlanar(w, u, bias, self.negative_slope)


class _UnconditionalPlanar(AbstractBijection):
    """Unconditional planar bijection, used in Planar.

    Note act_scale (u in the paper) is unconstrained and the constraint to ensure
    invertiblitiy is applied in ``get_act_scale``.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    weight: Array
    _act_scale: Array
    bias: Array
    activation: Literal["tanh"] | Literal["leaky_relu"]
    activation_fn: Callable
    negative_slope: float | None

    def __init__(
        self,
        weight: Float[Array, " dim"],
        act_scale: Float[Array, " dim"],
        bias: Float[Array, " "],
        negative_slope: float | None = None,
    ):
        self.weight = weight
        self.bias = bias
        self.shape = weight.shape
        self.negative_slope = negative_slope
        self._act_scale = act_scale

        if negative_slope is None:
            self.activation = "tanh"
            self.activation_fn = jnp.tanh
        else:
            if negative_slope <= 0:
                raise ValueError("The negative slope value should be >0.")
            self.activation = "leaky_relu"
            self.activation_fn = partial(nn.leaky_relu, negative_slope=negative_slope)

    def transform_and_log_det(self, x, condition=None):
        u = self.get_act_scale()
        act = self.activation_fn(x @ self.weight + self.bias)
        y = x + u * act
        if self.activation == "leaky_relu":
            psi = jnp.where(act < 0, self.negative_slope, 1) * self.weight
        else:
            psi = (1 - act**2) * self.weight
        log_det = jnp.log(jnp.abs(1 + u @ psi))
        return y, log_det

    def get_act_scale(self):
        """Apply constraint to u to ensure invertibility.

        See appendix A1 in https://arxiv.org/pdf/1505.05770.pdf.
        """
        wtu = self._act_scale @ self.weight
        m_wtu = -1 + jnp.log(1 + nn.softplus(wtu))
        return self._act_scale + (m_wtu - wtu) * self.weight / norm(self.weight) ** 2

    def inverse_and_log_det(self, y, condition=None):
        if self.activation != "leaky_relu":
            raise NotImplementedError(
                "The inverse planar transformation is only implemented with the leaky "
                "relu activation function.",
            )
        # Expanding explanation as the inverse is not in the original paper.
        # The derivation steps for the inversion are:
        # 1. Let z = w^Tx+b
        # 2. We want x=y-uσ(z), where σ is the leaky relu function.
        # 3. Sub x=y-uσ(z) into z = w^Tx+b,
        # 4. Solve for z, which gives z = (w^Ty+b)/(1+w^Tus), where s is the slope
        #   σ'(z), i.e. s=1 if z>=0 and s=negative_slope otherwise. To find the
        #   slope, it is sufficient to check the sign of the numerator w^Ty+b, rather
        #   than z, as the denominator is constrained to be positive.
        # 5. Compute inverse using x=y-uσ(z)

        numerator = self.weight @ y + self.bias
        relu_slope = jnp.where(numerator < 0, self.negative_slope, 1)
        us = self.get_act_scale() * relu_slope
        denominator = 1 + self.weight @ us
        log_det = -jnp.log(jnp.abs(1 + us @ self.weight))
        x = y - us * (numerator / denominator)
        return x, log_det
