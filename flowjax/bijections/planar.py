"""Planar transform.

A layer in a planar flow introduced in https://arxiv.org/pdf/1505.05770.pdf.
"""

from collections.abc import Callable
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax.nn import softplus
from jax.numpy.linalg import norm
from jaxtyping import Array, PRNGKeyArray

from flowjax.bijections.bijection import AbstractBijection


class Planar(AbstractBijection):
    r"""Planar bijection as used by https://arxiv.org/pdf/1505.05770.pdf.

    Uses the transformation :math:`y + u \cdot \text{tanh}(w \cdot x + b)`, where
    :math:`u \in \mathbb{R}^D, \ w \in \mathbb{R}^D` and :math:`b \in \mathbb{R}`. In
    the unconditional case, :math:`w`, :math:`u`  and :math:`b` are learned directly.
    In the conditional case they are parameterised by an MLP.

    Args:
        key: Jax random seed.
        dim: Dimension of the bijection.
        cond_dim: Dimension of extra conditioning variables. Defaults to None.
        **mlp_kwargs: Keyword arguments (excluding in_size and out_size) passed to
            the MLP (equinox.nn.MLP). Ignored when cond_dim is None.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    conditioner: Callable | None
    params: Array | None

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        dim: int,
        cond_dim: int | None = None,
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

    def transform(self, x, condition=None):
        return self.get_planar(condition).transform(x)

    def transform_and_log_det(self, x, condition=None):
        return self.get_planar(condition).transform_and_log_det(x)

    def inverse(self, y, condition=None):
        return self.get_planar(condition).inverse(y)

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
        return _UnconditionalPlanar(w, u, bias)


class _UnconditionalPlanar(AbstractBijection):
    """Unconditional planar bijection, used in Planar.

    Note act_scale (u in the paper) is unconstrained and the constraint to ensure
    invertiblitiy is applied in the ``get_act_scale``.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    weight: Array
    _act_scale: Array
    bias: Array

    def __init__(self, weight, act_scale, bias):
        self.weight = weight
        self._act_scale = act_scale
        self.bias = bias
        self.shape = weight.shape

    def transform(self, x, condition=None):
        return x + self.get_act_scale() * jnp.tanh(self.weight @ x + self.bias)

    def transform_and_log_det(self, x, condition=None):
        u = self.get_act_scale()
        act = jnp.tanh(x @ self.weight + self.bias)
        y = x + u * act
        psi = (1 - act**2) * self.weight
        log_det = jnp.log(jnp.abs(1 + u @ psi))
        return y, log_det

    def get_act_scale(self):
        """Apply constraint to u to ensure invertibility.

        See appendix A1 in https://arxiv.org/pdf/1505.05770.pdf.
        """
        wtu = self._act_scale @ self.weight
        m_wtu = -1 + jnp.log(1 + softplus(wtu))
        return self._act_scale + (m_wtu - wtu) * self.weight / norm(self.weight) ** 2

    def inverse(self, y, condition=None):
        raise NotImplementedError(
            "The inverse planar transformation is not implemented.",
        )

    def inverse_and_log_det(self, y, condition=None):
        raise NotImplementedError(
            "The inverse planar transformation is not implemented.",
        )
