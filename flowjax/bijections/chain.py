"""Bijections that chain together (compose) multiple bijections."""

from functools import partial
from typing import Any, Sequence, Tuple, Union

import equinox as eqx
from jax.lax import scan

from flowjax.bijections import Bijection
from flowjax.utils import Array


class Chain(Bijection):
    """Chain together arbitrary bijections to form another bijection."""

    bijections: Tuple[Bijection]
    cond_dim: int

    def __init__(self, bijections: Sequence[Bijection]):
        """
        Args:
            bijections (Sequence[Bijection]): Sequence of bijections.
        """
        self.bijections = tuple(bijections)
        self.cond_dim = max([b.cond_dim for b in bijections])

    def transform(self, x, condition=None):
        for bijection in self.bijections:
            x = bijection.transform(x, condition)
        return x

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        log_abs_det_jac = 0
        for bijection in self.bijections:
            x, log_abs_det_jac_i = bijection.transform_and_log_abs_det_jacobian(
                x, condition
            )
            log_abs_det_jac += log_abs_det_jac_i
        return x, log_abs_det_jac

    def inverse(self, y: Array, condition=None):
        for bijection in reversed(self.bijections):
            y = bijection.inverse(y, condition)
        return y

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        log_abs_det_jac = 0
        for bijection in reversed(self.bijections):
            y, log_abs_det_jac_i = bijection.inverse_and_log_abs_det_jacobian(
                y, condition
            )
            log_abs_det_jac += log_abs_det_jac_i
        return y, log_abs_det_jac

    def __getitem__(self, i: Union[int, slice]) -> Bijection:
        if isinstance(i, int):
            return self.bijections[i]
        elif isinstance(i, slice):
            return Chain(self.bijections[i])
        else:
            raise TypeError(f"Indexing with type {type(i)} is not supported.")

    def __iter__(self):
        yield from self.bijections

    def __len__(self):
        return len(self.bijections)


class ScannableChain(Bijection):
    """Repeatedly apply the same bijection with different parameter values. Internally,
    uses `jax.lax.scan` to reduce compilation time.

    Example:

        .. doctest::
            >>> from flowjax.bijections import Chain
            >>> import jax.numpy as jnp
            >>> import equinox as eqx

            >>> params = jnp.ones((3, 2))
            >>> # Below is equivilent to Chain([Affine(p) for p in params])
            >>> affine = ScannableChain(equinox.filter_vmap(Affine)(params))
    """
    static: Any
    params: Any
    cond_dim: int

    def __init__(self, bijection: Bijection):
        """
        The array leaves in `bijection` should have an additional leading axis to scan over.
        Often it is convenient to construct these using `equinox.filter_vmap`.

        
        Args:
            bijections (Bijection): A bijection, in which the arrays have an additional leading axis to scan over.
        """
        self.params, self.static = eqx.partition(bijection, eqx.is_array)  # type: ignore
        self.cond_dim = bijection.cond_dim

    def transform(self, x, condition=None):
        def fn(x, p, condition=None):
            bijection = eqx.combine(self.static, p)
            result = bijection.transform(x, condition)  # type: ignore
            return (result, None)

        fn = partial(fn, condition=condition)
        y, _ = scan(fn, x, self.params)
        return y

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        def fn(carry, p, condition):
            x, log_det = carry
            bijection = eqx.combine(self.static, p)
            y, log_det_i = bijection.transform_and_log_abs_det_jacobian(x, condition)  # type: ignore
            return ((y, log_det + log_det_i), None)

        fn = partial(fn, condition=condition)
        (y, log_det), _ = scan(fn, (x, 0), self.params)
        return y, log_det

    def inverse(self, y, condition=None):
        def fn(y, p, condition=None):
            bijection = eqx.combine(self.static, p)
            x = bijection.inverse(y, condition)  # type: ignore
            return (x, None)

        fn = partial(fn, condition=condition)
        x, _ = scan(fn, y, self.params, reverse=True)
        return x

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        def fn(carry, p, condition=None):
            y, log_det = carry
            bijection = eqx.combine(self.static, p)
            x, log_det_i = bijection.inverse_and_log_abs_det_jacobian(y, condition)  # type: ignore
            return ((x, log_det + log_det_i), None)

        fn = partial(fn, condition=condition)
        (y, log_det), _ = scan(fn, (y, 0), self.params, reverse=True)
        return y, log_det
