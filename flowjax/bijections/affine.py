"""Affine bijections."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import ClassVar

import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from flowjax import wrappers
from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.softplus import SoftPlus
from flowjax.utils import arraylike_to_array


class Affine(AbstractBijection):
    """Elementwise affine transformation ``y = a*x + b``.

    ``loc`` and ``scale`` should broadcast to the desired shape of the bijection.
    Note by default, we constrain the scale parameter to be postive using ``SoftPlus``.

    Args:
        loc: Location parameter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
        scale_constraint: Bijection with shape broadcastable to the shape of the scale
            parameter, used for reparameterization (with ``BijectionReparam``).
            Defaults to :class:`~flowjax.bijections.SoftPlus`.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    scale: Array | wrappers.AbstractUnwrappable[Array]

    def __init__(
        self,
        loc: ArrayLike = 0,
        scale: ArrayLike = 1,
        scale_constraint: AbstractBijection | None = None,
    ):
        scale_constraint = SoftPlus() if scale_constraint is None else scale_constraint
        self.loc, scale = jnp.broadcast_arrays(
            *(arraylike_to_array(a, dtype=float) for a in (loc, scale)),
        )
        self.shape = scale.shape
        self.scale = wrappers.BijectionReparam(scale, scale_constraint)

    def transform(self, x, condition=None):
        return x * self.scale + self.loc

    def transform_and_log_det(self, x, condition=None):
        return x * self.scale + self.loc, jnp.log(jnp.abs(self.scale)).sum()

    def inverse(self, y, condition=None):
        return (y - self.loc) / self.scale

    def inverse_and_log_det(self, y, condition=None):
        return (y - self.loc) / self.scale, -jnp.log(jnp.abs(self.scale)).sum()


class Loc(AbstractBijection):
    """Location transformation ``y = x + c``.

    Args:
        loc: Scale parameter. Defaults to 1.
    """

    loc: Array
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, loc: ArrayLike):
        self.loc = arraylike_to_array(loc)
        self.shape = self.loc.shape

    def transform(self, x, condition=None):
        return x + self.loc

    def transform_and_log_det(self, x, condition=None):
        return x + self.loc, jnp.zeros(())

    def inverse(self, y, condition=None):
        return y - self.loc

    def inverse_and_log_det(self, y, condition=None):
        return y - self.loc, jnp.zeros(())


class Scale(AbstractBijection):
    """Scale transformation ``y = a*x``.

    Args:
        scale: Scale parameter. Defaults to 1.
        scale_constraint: Bijection with shape broadcastable to the shape of the scale
            parameter, used for reparameterization. Defaults to
            :class:`~flowjax.bijections.SoftPlus`.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    scale: Array | wrappers.AbstractUnwrappable[Array]

    def __init__(
        self,
        scale: ArrayLike | wrappers.AbstractUnwrappable[Array],
        scale_constraint: AbstractBijection | None = None,
    ):
        scale_constraint = SoftPlus() if scale_constraint is None else scale_constraint
        self.scale = wrappers.BijectionReparam(scale, scale_constraint)
        self.shape = jnp.shape(wrappers.unwrap(scale))

    def transform(self, x, condition=None):
        return x * self.scale

    def transform_and_log_det(self, x, condition=None):
        return x * self.scale, jnp.log(jnp.abs(self.scale)).sum()

    def inverse(self, y, condition=None):
        return y / self.scale

    def inverse_and_log_det(self, y, condition=None):
        return y / self.scale, -jnp.log(jnp.abs(self.scale)).sum()


class TriangularAffine(AbstractBijection):
    r"""A triangular affine transformation.

    Transformation has the form :math:`Ax + b`, where :math:`A` is a lower or upper
    triangular matrix, and :math:`b` is the bias vector.

    Args:
        loc: Location parameter. If this is scalar, it is broadcast to the dimension
            inferred from arr.
        arr: Triangular matrix.
        lower: Whether the mask should select the lower or upper
            triangular matrix (other elements ignored). Defaults to True (lower).
        diag_constraint: Bijection with shape broadcastable to the shape of the
            bijection, used for reparameterization of the diagonal elements. Defaults to
            :class:`~flowjax.bijections.SoftPlus`.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    triangular: Array | wrappers.AbstractUnwrappable[Array]
    lower: bool

    def __init__(
        self,
        loc: ArrayLike,
        arr: ArrayLike,
        *,
        lower: bool = True,
        diag_constraint: AbstractBijection | None = None,
    ):
        diag_constraint = SoftPlus() if diag_constraint is None else diag_constraint
        loc, arr = (arraylike_to_array(a, dtype=float) for a in (loc, arr))
        if (arr.ndim != 2) or (arr.shape[0] != arr.shape[1]):
            raise ValueError("arr must be a square, 2-dimensional matrix.")

        dim = arr.shape[0]

        # Make triangular parameterization by adding diagonal and triangular matrices.
        if lower:
            tri_mask = jnp.tril(jnp.ones((dim, dim), dtype=jnp.int32), k=-1)
        else:
            tri_mask = jnp.triu(jnp.ones((dim, dim), dtype=jnp.int32), k=1)

        triangular = wrappers.Lambda(
            lambda tril, diag: sum([tril, diag]),
            tril=wrappers.Where(tri_mask, arr, 0),
            diag=wrappers.Diagonal(
                wrappers.BijectionReparam(jnp.diag(arr), diag_constraint)
            ),
        )
        self.lower = lower
        self.shape = (dim,)
        self.triangular = triangular
        self.loc = jnp.broadcast_to(loc, (dim,))

    def transform(self, x, condition=None):
        return self.triangular @ x + self.loc

    def transform_and_log_det(self, x, condition=None):
        y = self.triangular @ x + self.loc
        return y, jnp.log(jnp.abs(jnp.diag(self.triangular))).sum()

    def inverse(self, y, condition=None):
        return solve_triangular(self.triangular, y - self.loc, lower=self.lower)

    def inverse_and_log_det(self, y, condition=None):
        x = solve_triangular(self.triangular, y - self.loc, lower=self.lower)
        return x, -jnp.log(jnp.abs(jnp.diag(self.triangular))).sum()


class AdditiveCondition(AbstractBijection):
    """Given a callable ``f``, carries out the transformation ``y = x + f(condition)``.

    If used to transform a distribution, this allows the "location" to be changed as a
    function of the conditioning variables. Note that the callable can be a callable
    module with trainable parameters.

    Args:
        module: A callable (e.g. a function or callable module) that maps array with
            shape cond_shape, to a shape that is broadcastable with the shape of the
            bijection.
        shape: The shape of the bijection.
        cond_shape: The condition shape of the bijection.

    Example:
        Conditioning using a linear transformation

        .. doctest::

            >>> from flowjax.bijections import AdditiveCondition
            >>> from equinox.nn import Linear
            >>> import jax.numpy as jnp
            >>> import jax.random as jr
            >>> bijection = AdditiveCondition(
            ...     Linear(2, 3, key=jr.PRNGKey(0)), shape=(3,), cond_shape=(2,)
            ...     )
            >>> bijection.transform(jnp.ones(3), condition=jnp.ones(2))
            Array([1.9670618, 0.8156546, 1.7763454], dtype=float32)

    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    module: Callable[[ArrayLike], ArrayLike]

    def __init__(
        self,
        module: Callable[[ArrayLike], ArrayLike],
        shape: tuple[int, ...],
        cond_shape: tuple[int, ...],
    ):
        self.module = module
        self.shape = shape
        self.cond_shape = cond_shape

    def transform(self, x, condition=None):
        return x + self.module(condition)

    def transform_and_log_det(self, x, condition=None):
        return self.transform(x, condition), jnp.array(0)

    def inverse(self, y, condition=None):
        return y - self.module(condition)

    def inverse_and_log_det(self, y, condition=None):
        return self.inverse(y, condition), jnp.array(0)
