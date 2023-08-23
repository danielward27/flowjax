"""Abstact base classes for the `Bijection` and `Bijection` types. Note when
implementing bijections, by convention we try to i) implement the "transform" methods as
the faster/more intuitive approach (compared to the inverse methods); and ii) implement
only the forward methods if an inverse is not available. The `Invert` bijection can be
used to invert the orientation if a fast inverse is desired (e.g. maximum likelihood
fitting of flows).
"""

from abc import abstractmethod

from equinox import Module
from jax import Array
from jax.typing import ArrayLike

from flowjax.utils import arraylike_to_array


class Bijection(Module):
    """Bijection base class. Similar to :py:class:`~flowjax.distributions.Distribution`,
    bijections have a ``shape`` and a ``cond_shape`` attribute. To allow easy composing
    of bijections, all bijections support passing of conditioning variables (even if
    ignored).

    The methods of bijections do not generally support passing of additional batch
    dimensions, however, ``jax.vmap`` or ``eqx.filter_vmap`` can be used to vmap
    specific methods if desired, and a bijection can be explicitly vectorised using the
    :py:class:`~flowjax.bijections.jax_transforms.Batch` bijection.

    Bijections are registered as Jax PyTrees (as they are equinox modules), so are
    compatible with normal jax operations.

    **Implementing a bijection**

        (1) Inherit from ``Bijection``.
        (2) Define the attributes ``shape`` and ``cond_shape``. A ``cond_shape`` of
            ``None`` is used to represent unconditional bijections.
        (3) Implement the abstract methods ``transform``, ``transform_and_log_det``,
            ``inverse`` and ``inverse_and_log_det``. These should act on
            inputs compatible with the shapes ``shape`` for ``x``, and ``cond_shape``
            for ``condition``.

    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None

    @abstractmethod
    def transform(self, x: ArrayLike, condition: ArrayLike | None = None) -> Array:
        """Apply transformation."""

    @abstractmethod
    def transform_and_log_det(
        self, x: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        """Apply transformation and compute log absolute value of the Jacobian
        determinant."""

    @abstractmethod
    def inverse(self, y: ArrayLike, condition: ArrayLike | None = None) -> Array:
        """Invert the transformation."""

    @abstractmethod
    def inverse_and_log_det(
        self, y: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        """Invert the transformation and compute log absolute value of the Jacobian
        determinant."""

    def _argcheck_and_cast(
        self, x: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array | None]:
        """Utility function that checks input shapes against the bijection shapes,
        and casts inputs to arrays if required. Note this permits passing a condition
        in the case when bijection.cond_shape is None."""
        x = arraylike_to_array(x, err_name="x")

        if x.shape != self.shape:
            raise ValueError(f"Expected x.shape {self.shape}; got {x.shape}")

        if condition is not None:
            condition = arraylike_to_array(condition, err_name="condition")

            if self.cond_shape is not None and condition.shape != self.cond_shape:
                raise ValueError(
                    f"Expected condition.shape {self.cond_shape}; got {condition.shape}"
                )

        return x, condition
