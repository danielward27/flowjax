"""Abstact base classes for the `Bijection` and `Bijection` types.

Note when implementing bijections, by convention we try to i) implement the "transform"
methods as the faster/more intuitive approach (compared to the inverse methods); and ii)
implement only the forward methods if an inverse is not available. The `Invert`
bijection can be used to invert the orientation if a fast inverse is desired (e.g.
maximum likelihood fitting of flows).
"""

import functools
import inspect
from abc import abstractmethod
from contextlib import suppress

import equinox as eqx
from equinox import AbstractVar, Module
from jax import Array
from jax.typing import ArrayLike

from flowjax.utils import arraylike_to_array


class _BijectionMeta(type(eqx.Module)):
    # Metaclass for bijections. This serves two roles:
    # 1) wraps methods to enforce input shapes to match the shapes of the bijection.
    # 2) Converts inputs to Array where appropriate.
    def __new__(
        mcs,
        name,
        bases,
        dict_,
        /,
        strict: bool = False,
        abstract: bool = False,
        **kwargs,
    ):
        wrap_methods = [
            "transform",
            "transform_and_log_det",
            "inverse",
            "inverse_and_log_det",
        ]
        with suppress(NameError):
            if AbstractBijection in bases and not inspect.isabstract(mcs):
                for meth in wrap_methods:
                    if not hasattr(dict_[meth], "__isabstractmethod__"):
                        dict_[meth] = _check_and_cast(dict_[meth])
        return super().__new__(mcs, name, bases, dict_, strict, abstract, **kwargs)


class AbstractBijection(Module, metaclass=_BijectionMeta):
    """Bijection abstract class.

    Similar to :py:class:`~flowjax.distributions.AbstractDistribution`, bijections have
    a ``shape`` and a ``cond_shape`` attribute. To allow easy composing of bijections,
    all bijections support passing of conditioning variables (even if ignored).

    The methods of bijections do not generally support passing of additional batch
    dimensions, however, ``jax.vmap`` or ``eqx.filter_vmap`` can be used to vmap
    specific methods if desired, and a bijection can be explicitly vectorised using the
    :class:`~flowjax.bijections.jax_transforms.Vmap` bijection.

    Bijections are registered as Jax PyTrees (as they are equinox modules), so are
    compatible with normal jax operations.

    **Implementing a bijection**

        (1) Inherit from ``AbstractBijection``.
        (2) Define the attributes ``shape`` and ``cond_shape``. A ``cond_shape`` of
            ``None`` is used to represent unconditional bijections.
        (3) Implement the abstract methods ``transform``, ``transform_and_log_det``,
            ``inverse`` and ``inverse_and_log_det``. These should act on
            inputs compatible with the shapes ``shape`` for ``x``, and ``cond_shape``
            for ``condition``.
    """

    shape: AbstractVar[tuple[int, ...]]
    cond_shape: AbstractVar[tuple[int, ...] | None]

    @abstractmethod
    def transform(self, x: ArrayLike, condition: ArrayLike | None = None) -> Array:
        """Apply the forward transformation.

        Args:
            x (ArrayLike): Input with shape matching bijections.shape.
            condition (ArrayLike | None, optional): Condition, with shape matching
                bijection.cond_shape, required for conditional bijections. Defaults to
                None.
        """

    @abstractmethod
    def transform_and_log_det(
        self,
        x: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        """Apply transformation and compute the log absolute Jacobian determinant.

        Args:
            x (ArrayLike): Input with shape matching the bijections shape
            condition (ArrayLike | None, optional): . Defaults to None.
        """

    @abstractmethod
    def inverse(self, y: ArrayLike, condition: ArrayLike | None = None) -> Array:
        """Compute the inverse transformation.

        Args:
            y (ArrayLike): Input array with shape matching bijection.shape
            condition (ArrayLike | None, optional): Condition array with shape matching
                bijection.cond_shape. Required for conditional bijections. Defaults to
                None.
        """

    @abstractmethod
    def inverse_and_log_det(
        self,
        y: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        """Inverse transformation and corresponding log absolute jacobian determinant.

        Args:
            y (ArrayLike): Input array with shape matching bijection.shape.
            condition (ArrayLike | None, optional): Condition array with shape matching
                bijection.cond_shape. Required for conditional bijections. Defaults to
                None.
        """


def _check_and_cast(method):
    """Decorator that performs argument checking and converts arraylike to array."""

    @functools.wraps(method)
    def wrapper(self, x, condition=None):
        def _check_condition(condition):
            if condition is not None:
                condition = arraylike_to_array(condition, err_name="condition")

            if self.cond_shape is not None and condition.shape != self.cond_shape:
                raise ValueError(
                    f"Expected condition.shape {self.cond_shape}; got {condition.shape}",
                )
            return condition

        def _check_x(x):
            x = arraylike_to_array(x)
            if x.shape != self.shape:
                raise ValueError(f"Expected input shape {self.shape}; got {x.shape}")
            return x

        return method(self, _check_x(x), _check_condition(condition))

    return wrapper
