""""Bisection search algorithm.

This is useful for inverting some bijections that do not have a known inverse.
"""

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jax import lax


class AutoregressiveBisectionInverter(eqx.Module):
    """Callable module to invert an autoregressive bijection using a bisection search.

    Note that if the value is not within lower and upper, the bounds are dynamically
    adjusted using ``adapt_interval_to_include_root``.

    Args:
        lower: Lower bound of the initial interval where the inverse value is expected.
        upper: Upper bound of the initial interval where the inverse value is expected.
        tol: Tolerance of solution. Note due to accumulation of errors at each step,
            the found solution may not fall within the given tolerance.
        max_iter: Maximum number of iterations to use.
    """

    lower: float = -10.0
    upper: float = 10.0
    tol: float = 1e-7
    max_iter: int = 200

    def __check_init__(self):
        if self.lower >= self.upper:
            raise ValueError("Lower must be less than upper.")
        if self.tol <= 0:
            raise ValueError("Tolerance must be positive.")

    def __call__(self, bijection, y, condition=None):
        def fn(x):
            return bijection.transform(x, condition) - y

        return autoregressive_bisection_search(
            autoregressive_fn=fn,
            lower=self.lower,
            upper=self.upper,
            tol=self.tol,
            length=bijection.shape[0],
            max_iter=self.max_iter,
        )


def autoregressive_bisection_search(
    autoregressive_fn: Callable,
    *,
    lower: float,
    upper: float,
    tol: float,
    length: int,
    max_iter: int,
):
    """Bisection search for a monotonic increasing autoregressive function.

    We scan over the inputs finding the root element by element, assuming that
    each input only depends on previous inputs in the input array. This is useful
    for inverting some bijections without a known inverse, such as those used in block
    neural autoregressive flows. Note that tol refers to the tolerance used in each
    run of the bisection search. This means the found solution may not necessarily
    be within the tolerance for all elements, as errors can accumulate in each step.

    Args:
        autoregressive_fn: The monotonically increasing autoregressive function.
        length: The length of the vector input to the function.
        tol: Tolerance of a solution. Note due to accumulation of errors at each step,
            the found solution may not fall within the given tolerance.
        lower: Lower bound of the initial interval where the root is expected.
        upper: Upper bound of the initial interval where the root is expected.
        max_iter: Maximum number of iterations. Defaults to 200.
    """

    def scan_fn(init, _):
        y, i = init

        def scalar_fn(x):
            x = y.at[i].set(x)
            return autoregressive_fn(x)[i]

        root, *_ = bisection_search(
            scalar_fn,
            tol=tol,
            lower=lower,
            upper=upper,
            max_iter=max_iter,
        )

        return (y.at[i].set(root), i + 1), None

    init = (jnp.full(length, (upper + lower) / 2), 0)
    (root, _), _ = lax.scan(f=scan_fn, init=init, xs=None, length=length)
    return root


def bisection_search(
    func: Callable,
    *,
    lower: float,
    upper: float,
    tol: float,
    max_iter: int,
):
    """Bisection search algorithm to find a root of a scalar increasing function.

    Note that the implementation assumes that the function is monotonically increasing.
    If the initial limits do not include the root then the interval is adapted using
    ``expand_interval_to_include_root``.

    Args:
        func: Scalar increasing function to find the root for.
        lower: Lower bound of the initial interval where the root is expected.
        upper: Upper bound of the initial interval where the root is expected.
        tol: Tolerance level for the root. Defaults to 1e-6. Defaults to 1e-6.
        max_iter: Maximum number of iterations. Defaults to 200.

    Returns:
        A tuple ``(root, adapt_iterations, iterations)``. ``adapt_iterations`` will be
            zero if the root is beteween lower and upper).
    """
    lower, upper, adapt_iterations = adapt_interval_to_include_root(
        func,
        lower=lower,
        upper=upper,
    )

    def cond_fn(state):
        lower, upper, iterations = state
        return jnp.logical_and((upper - lower) > 2 * tol, iterations < max_iter)

    def body_fn(state):
        lower, upper, iterations = state
        midpoint = (lower + upper) / 2
        is_result_positive = func(midpoint) >= 0
        lower = jnp.where(is_result_positive, lower, midpoint)
        upper = jnp.where(is_result_positive, midpoint, upper)
        return lower, upper, iterations + 1

    init_state = (lower, upper, 0)
    lower, upper, iterations = lax.while_loop(cond_fn, body_fn, init_state)
    root = (lower + upper) / 2
    return root, adapt_iterations, iterations


def adapt_interval_to_include_root(
    func,
    *,
    lower: float,
    upper: float,
    expand_factor: float = 2.0,
    max_iterations: int = 200,
):
    """Dyamically adjust the interval to include the root of an increasing function.

    Note we do not currently perform any argument checking as it is challenging to
    perform checks that rely on array values with jax. It is the users responsibility
    to ensure lower is less than upper, and the function is increasing.

    Args:
        func: A scalar increasing function.
        lower: Lower component of interval.
        upper: Upper component of interval.
        expand_factor: How much to (multiplicatively) increase the adjustment by on
            each iteration. The magnitude of the adjustment of the necessary bound is
            given by ``(init_upper - init_lower)*expand_factor**iteration``. Defaults to
            2.0.
        max_iterations: The maxmimum number of iterations before the function errors.
            Defaults to 100.
    """
    fn_lower, fn_upper = func(lower), func(upper)

    class _State(NamedTuple):
        lower: float
        upper: float
        expand_by: float
        lower_fn_sign: int
        upper_fn_sign: int
        iteration: int = 0

    def cond_fn(state):
        signs_match = state.lower_fn_sign == state.upper_fn_sign
        return jnp.logical_and(signs_match, state.iteration < max_iterations)

    def body_fn(state):
        sign = state.lower_fn_sign  # Note we know the signs match from cond_fn

        return _State(
            lower=jnp.where(sign == 1, state.lower - state.expand_by, state.upper),
            upper=jnp.where(sign == 1, state.lower, state.upper + state.expand_by),
            expand_by=state.expand_by * expand_factor,
            lower_fn_sign=jnp.sign(func(state.lower)),
            upper_fn_sign=jnp.sign(func(state.upper)),
            iteration=state.iteration + 1,
        )

    lower, upper = jnp.asarray(lower, float), jnp.asarray(upper, float)

    init_state = _State(
        lower=lower,
        upper=upper,
        expand_by=upper - lower,
        lower_fn_sign=jnp.sign(fn_lower),
        upper_fn_sign=jnp.sign(fn_upper),
    )
    state = lax.while_loop(cond_fn, body_fn, init_state)

    return state.lower, state.upper, state.iteration
