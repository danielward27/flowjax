"""Abstact base classes for the `Bijection` and `Bijection` types.

Note when implementing bijections, by convention we try to i) implement the "transform"
methods as the faster/more intuitive approach (compared to the inverse methods); and ii)
implement only the forward methods if an inverse is not available. The `Invert`
bijection can be used to invert the orientation if a fast inverse is desired (e.g.
maximum likelihood fitting of flows).
"""

import functools
from abc import abstractmethod

import equinox as eqx
from jax import Array

import flowjax
from flowjax._custom_types import ArrayLike


def _unwrap_check_and_cast(method):
    """Decorator that unwraps unwrappables, performs argument casting and checking."""

    @functools.wraps(method)
    def wrapper(bijection, x, condition=None):

        def _check_condition(condition):
            if condition is not None:
                condition = flowjax.utils.arraylike_to_array(
                    condition, err_name="condition"
                )
            elif bijection.cond_shape is not None:
                raise ValueError("Expected condition to be provided.")

            if (
                bijection.cond_shape is not None
                and condition.shape != bijection.cond_shape
            ):
                raise ValueError(
                    f"Expected condition.shape {bijection.cond_shape}; got "
                    f"{condition.shape}",
                )
            return condition

        def _check_x(x):
            x = flowjax.utils.arraylike_to_array(x)
            if x.shape != bijection.shape:
                raise ValueError(
                    f"Expected input shape {bijection.shape}; got {x.shape}"
                )
            return x

        return method(
            flowjax.wrappers.unwrap(bijection), _check_x(x), _check_condition(condition)
        )

    wrapper.__unwrap_check_and_cast__ = True
    return wrapper


class AbstractBijection(eqx.Module):
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

    shape: eqx.AbstractVar[tuple[int, ...]]
    cond_shape: eqx.AbstractVar[tuple[int, ...] | None]

    def __init_subclass__(cls) -> None:
        # We wrap the class methods with argument checking
        wrap_methods = [
            "transform",
            "transform_and_log_det",
            "inverse",
            "inverse_and_log_det",
        ]
        for meth in wrap_methods:
            if meth in cls.__dict__ and not hasattr(
                cls.__dict__[meth],
                "__isabstractmethod__",
            ):
                assert not hasattr(cls.__dict__[meth], "__unwrap_check_and_cast__")
                setattr(cls, meth, _unwrap_check_and_cast(cls.__dict__[meth]))
        return cls

    @abstractmethod
    def transform(self, x: ArrayLike, condition: ArrayLike | None = None) -> Array:
        """Apply the forward transformation.

        Args:
            x: Input with shape matching ``bijections.shape``.
            condition: Condition, with shape matching ``bijection.cond_shape``, required
                for conditional bijections and ignored for unconditional bijections.
                Defaults to None.
        """

    @abstractmethod
    def transform_and_log_det(
        self,
        x: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        """Apply transformation and compute the log absolute Jacobian determinant.

        Args:
            x: Input with shape matching the bijections shape
            condition: . Defaults to None.
        """

    @abstractmethod
    def inverse(self, y: ArrayLike, condition: ArrayLike | None = None) -> Array:
        """Compute the inverse transformation.

        Args:
            y: Input array with shape matching bijection.shape
            condition: Condition array with shape matching bijection.cond_shape.
                Required for conditional bijections. Defaults to None.
        """

    @abstractmethod
    def inverse_and_log_det(
        self,
        y: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        """Inverse transformation and corresponding log absolute jacobian determinant.

        Args:
            y: Input array with shape matching bijection.shape.
            condition: Condition array with shape matching bijection.cond_shape.
                Required for conditional bijections. Defaults to None.
        """
