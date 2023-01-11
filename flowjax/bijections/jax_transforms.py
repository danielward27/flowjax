from functools import partial
import equinox as eqx
from flowjax.bijections import Bijection
from jax.lax import scan
from typing import Any, Tuple


class Vmap(Bijection):
    """Expand the dimension of a bijection by vmapping. By default, we vmap over
    bijection parameters, x and the conditioning variables, although this behaviour can
    be modified by providing key word arguments that are passed to eqx.filter_vmap.

     Example:

        .. doctest::
            >>> # TODO create examples

    """

    ndim_to_add: int
    bijection: Bijection
    kwargs: dict

    def __init__(self, bijection: Bijection, shape: Tuple[int], **kwargs):
        """
        Args:
            bijection (Bijection): Bijection, where the parameters have leading shape equalling shape (if vmapping over bijection parameters).
            shape (Tuple[int]): Shape prepended to the bijection shape. If len(shape)>1, multiple applications of vmap will be used.
            **kwargs: kwargs, passed to equinox.filter_vmap, allowing e.g. control over which variables to map over.
        """
        self.bijection = bijection
        self.ndim_to_add = len(shape)
        self.kwargs = kwargs  # For filter vmap
        self.shape = (
            shape + self.bijection.shape if self.bijection.shape is not None else shape
        )
        self.cond_shape = bijection.cond_shape

    def transform(self, x, condition=None):
        self._argcheck(x, condition)
        f = lambda b, x, c: b.transform(x, c)
        f = self._multivmap(f)
        return f(self.bijection, x, condition)

    def inverse(self, y, condition=None):
        self._argcheck(y, condition)
        f = lambda b, y, c: b.inverse(y, c)
        f = self._multivmap(f)
        return f(self.bijection, y, condition)

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        self._argcheck(x, condition)
        f = lambda b, x, c: b.transform_and_log_abs_det_jacobian(x, c)
        f = self._multivmap(f)
        y, log_det = f(self.bijection, x, condition)
        return y, log_det.sum()

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        self._argcheck(y, condition)
        f = lambda b, y, c: b.inverse_and_log_abs_det_jacobian(y, c)
        f = self._multivmap(f)
        x, log_det = f(self.bijection, y, condition)
        return x, log_det.sum()

    def _multivmap(self, f):
        "Compose Vmap to add ndim batch dimensions."
        for _ in range(self.ndim_to_add):
            f = eqx.filter_vmap(f, **self.kwargs)
        return f


class Scan(Bijection):
    """Repeatedly apply the same bijection with different parameter values. Internally,
    uses `jax.lax.scan` to reduce compilation time.

    Example:

        .. doctest::
            >>> from flowjax.bijections import Chain
            >>> import jax.numpy as jnp
            >>> import equinox as eqx

            >>> params = jnp.ones((3, 2))
            >>> # Below is equivilent to Chain([Affine(p) for p in params])
            >>> affine = Scan(equinox.filter_vmap(Affine)(params))
    """

    static: Any
    params: Any

    def __init__(self, bijection: Bijection):
        """
        The array leaves in `bijection` should have an additional leading axis to scan over.
        Often it is convenient to construct these using `equinox.filter_vmap`.


        Args:
            bijections (Bijection): A bijection, in which the arrays have an additional leading axis to scan over.
        """
        self.params, self.static = eqx.partition(bijection, eqx.is_array)  # type: ignore
        self.shape = bijection.shape
        self.cond_shape = bijection.cond_shape

    def transform(self, x, condition=None):
        self._argcheck(x, condition)

        def fn(x, p, condition=None):
            bijection = eqx.combine(self.static, p)
            result = bijection.transform(x, condition)  # type: ignore
            return (result, None)

        fn = partial(fn, condition=condition)
        y, _ = scan(fn, x, self.params)
        return y

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        self._argcheck(x, condition)

        def fn(carry, p, condition):
            x, log_det = carry
            bijection = eqx.combine(self.static, p)
            y, log_det_i = bijection.transform_and_log_abs_det_jacobian(x, condition)  # type: ignore
            return ((y, log_det + log_det_i.sum()), None)

        fn = partial(fn, condition=condition)
        (y, log_det), _ = scan(fn, (x, 0), self.params)
        return y, log_det

    def inverse(self, y, condition=None):
        self._argcheck(y, condition)

        def fn(y, p, condition=None):
            bijection = eqx.combine(self.static, p)
            x = bijection.inverse(y, condition)  # type: ignore
            return (x, None)

        fn = partial(fn, condition=condition)
        x, _ = scan(fn, y, self.params, reverse=True)
        return x

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        self._argcheck(y, condition)

        def fn(carry, p, condition=None):
            y, log_det = carry
            bijection = eqx.combine(self.static, p)
            x, log_det_i = bijection.inverse_and_log_abs_det_jacobian(y, condition)  # type: ignore
            return ((x, log_det + log_det_i.sum()), None)

        fn = partial(fn, condition=condition)
        (y, log_det), _ = scan(fn, (y, 0), self.params, reverse=True)
        return y, log_det