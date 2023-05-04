"""Bijections that wrap jax function transforms (scan and vmap)."""
from functools import partial
from typing import Any, Callable

import equinox as eqx
from jax.lax import scan
import jax.numpy as jnp
from flowjax.bijections.bijection import Bijection


class Batch(Bijection):
    """Add batch dimensions to a bijection, such that the new shape is
    batch_shape + bijection.shape. The batch dimensions are added using multiple
    applications of eqx.filter_vmap.

    Example:

    .. doctest::

        >>> import jax.numpy as jnp
        >>> from flowjax.bijections import Batch, Affine
        >>> x = jnp.ones(2)
        >>> batched = Batch(Affine(1), (2,), vectorize_bijection=False)
        >>> batched.transform(x)
        Array([2., 2.], dtype=float32)
    """

    bijection: Bijection
    in_axes: tuple
    batch_shape: tuple[int, ...]

    def __init__(
        self,
        bijection: Bijection,
        batch_shape: tuple[int, ...],
        vectorize_bijection: bool,
        vectorize_condition: bool | None = None,
    ):
        """
        Args:
            bijection (Bijection): Bijection to add batch dimensions to.
            batch_shape (tuple[int, ...]): The shape of the batch dimension.
            vectorize_bijection (bool): Whether to vectorise bijection parameters.
                * If True: we vectorize across the leading dimensions in the array
                leaves of the bijection. In this case, the array leaves must
                have leading dimensions equal to batch_shape. For construction of
                compatible bijections, see eqx.filter_vmap.
                * If False: we broadcast the parameters, meaning
                the same bijection parameters are used for each x.
            vectorize_condition (bool | None): Whether to vectorize or broadcast the
                conditioning variables. If broadcasting, the condition shape is
                unchanged. If vectorising, the condition shape will be
                ``batch_shape + bijection.cond_shape``. Defaults to None.
        """
        if vectorize_condition is None and bijection.cond_shape is not None:
            raise ValueError(
                "vectorize_condition must be specified for conditional bijections."
            )

        self.in_axes = (
            eqx.if_array(axis=0) if vectorize_bijection else None,
            eqx.if_array(axis=0),
            0 if vectorize_condition else None,
        )
        self.shape = batch_shape + bijection.shape
        self.batch_shape = batch_shape
        self.bijection = bijection

        if self.bijection.cond_shape is None:
            self.cond_shape = None
        elif vectorize_condition:
            self.cond_shape = batch_shape + self.bijection.cond_shape
        else:
            self.cond_shape = self.bijection.cond_shape

    def transform(self, x, condition=None):
        self._argcheck(x, condition)

        def _transform(bijection, x, condition):
            return bijection.transform(x, condition)

        return self.multi_vmap(_transform)(self.bijection, x, condition)

    def transform_and_log_det(self, x, condition=None):
        self._argcheck(x, condition)

        def _transform_and_log_det(bijection, x, condition):
            return bijection.transform_and_log_det(x, condition)

        y, log_det = self.multi_vmap(_transform_and_log_det)(
            self.bijection, x, condition
        )
        return y, jnp.sum(log_det)

    def inverse(self, y, condition=None):
        self._argcheck(y, condition)

        def _inverse(bijection, x, condition):
            return bijection.inverse(x, condition)

        return self.multi_vmap(_inverse)(self.bijection, y, condition)

    def inverse_and_log_det(self, y, condition=None):
        self._argcheck(y, condition)

        def _inverse_and_log_det(bijection, x, condition):
            return bijection.inverse_and_log_det(x, condition)

        x, log_det = self.multi_vmap(_inverse_and_log_det)(self.bijection, y, condition)
        return x, jnp.sum(log_det)

    def multi_vmap(self, func: Callable) -> Callable:
        """Compose vmap to add ndim batch dimensions."""
        for _ in range(len(self.batch_shape)):
            func = eqx.filter_vmap(func, in_axes=self.in_axes)
        return func


class Scan(Bijection):
    """Repeatedly apply the same bijection with different parameter values. Internally,
    uses `jax.lax.scan` to reduce compilation time.
    """

    static: Any
    params: Any

    def __init__(self, bijection: Bijection):
        """
        The array leaves in `bijection` should have an additional leading axis to scan over.
        Often it is convenient to construct these using `equinox.filter_vmap`.

        Args:
            bijection (Bijection): A bijection, in which the arrays leaves have an
                additional leading axis to scan over. For complex bijections, it can be
                convenient to create compatible bijections with ``equinox.filter_vmap``.

        Example:
            Below is equivilent to ``Chain([Affine(p) for p in params])``.

            .. doctest::

                >>> from flowjax.bijections import Scan, Affine
                >>> import jax.numpy as jnp
                >>> import equinox as eqx
                >>> params = jnp.ones((3, 2))
                >>> affine = Scan(Affine(params))

        """
        self.params, self.static = eqx.partition(bijection, eqx.is_array)  # type: ignore
        self.shape = bijection.shape
        self.cond_shape = bijection.cond_shape

    def transform(self, x, condition=None):
        self._argcheck(x, condition)

        def step(x, params, condition=None):
            bijection = eqx.combine(self.static, params)
            result = bijection.transform(x, condition)  # type: ignore
            return (result, None)

        step = partial(step, condition=condition)
        y, _ = scan(step, x, self.params)
        return y

    def transform_and_log_det(self, x, condition=None):
        self._argcheck(x, condition)

        def step(carry, params, condition):
            x, log_det = carry
            bijection = eqx.combine(self.static, params)
            y, log_det_i = bijection.transform_and_log_det(x, condition)  # type: ignore
            return ((y, log_det + log_det_i.sum()), None)

        step = partial(step, condition=condition)
        (y, log_det), _ = scan(step, (x, 0), self.params)
        return y, log_det

    def inverse(self, y, condition=None):
        self._argcheck(y, condition)

        def step(y, params, condition):
            bijection = eqx.combine(self.static, params)
            x = bijection.inverse(y, condition)  # type: ignore
            return (x, None)

        step = partial(step, condition=condition)
        x, _ = scan(step, y, self.params, reverse=True)
        return x

    def inverse_and_log_det(self, y, condition=None):
        self._argcheck(y, condition)

        def step(carry, params, condition):
            y, log_det = carry
            bijection = eqx.combine(self.static, params)
            x, log_det_i = bijection.inverse_and_log_det(y, condition)  # type: ignore
            return ((x, log_det + log_det_i.sum()), None)

        step = partial(step, condition=condition)
        (y, log_det), _ = scan(step, (y, 0), self.params, reverse=True)
        return y, log_det
