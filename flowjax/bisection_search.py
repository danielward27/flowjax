""""Bisection search algorithm.

This is useful for inverting some bijections that do not have a known inverse.
"""

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Int, Real


class AutoregressiveBisectionInverter(eqx.Module):
    """Callable module to invert an autoregressive bijection using a bisection search.

    Note that if the inverse value is not within lower and upper, the bounds are
    dynamically adjusted using ``_adapt_interval_to_include_root``.

    Args:
        lower: Lower bound of the initial interval where the inverse value is expected.
        upper: Upper bound of the initial interval where the inverse value is expected.
        tol: Tolerance of solution. Note due to accumulation of errors at each step,
            the found solution may not fall within the given tolerance.
        max_iter: Maximum number of iterations to use.
    """

    lower: float | int = -10
    upper: float | int = 10
    tol: float = 1e-7
    max_iter: int = 200

    def __check_init__(self):
        if not self.lower < self.upper:
            raise ValueError("Lower must be less than upper.")
        if self.tol <= 0:
            raise ValueError("Tolerance must be positive.")
        if self.max_iter < 0:
            raise ValueError("max_iter must be a positive integer.")

    def __call__(self, bijection, y, condition=None):
        def fn(x):
            return bijection.transform(x, condition) - y

        return _autoregressive_bisection_search(
            autoregressive_fn=fn,
            lower=jnp.array(self.lower, dtype=float),
            upper=jnp.array(self.upper, dtype=float),
            tol=self.tol,
            length=bijection.shape[0],
            max_iter=self.max_iter,
        )


def _autoregressive_bisection_search(
    autoregressive_fn: Callable,
    *,
    lower: Real[Array, ""],
    upper: Real[Array, ""],
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

        root, *_ = _bisection_search(
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


def _bisection_search(
    func: Callable,
    *,
    lower: Real[Array, ""],
    upper: Real[Array, ""],
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
    if max_iter < 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    lower, upper, adapt_iterations = _adapt_interval_to_include_root(
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
        sign = jnp.sign(func(midpoint))
        lower = jnp.where(sign == 1, lower, midpoint)
        upper = jnp.where(sign == 1, midpoint, upper)

        # In case we hit the root exactly
        lower = jnp.where(sign == 0, midpoint, lower)
        upper = jnp.where(sign == 0, midpoint, upper)
        return lower, upper, iterations + 1

    init_state = (lower, upper, 0)
    lower, upper, iterations = lax.while_loop(cond_fn, body_fn, init_state)
    root = (lower + upper) / 2
    return root, adapt_iterations, iterations


def _adapt_interval_to_include_root(
    func,
    *,
    lower: Real[Array, ""],
    upper: Real[Array, ""],
    expand_factor: float = 2.0,
):
    """Dyamically adjust the interval to include the root of an increasing function.

    Note we do not currently perform any argument checking as it is challenging to
    perform checks that rely on array values with JAX. It is the users responsibility
    to ensure lower is less than upper, and the function is increasing.

    Args:
        func: A scalar increasing function.
        lower: Lower component of interval. Must be less than upper.
        upper: Upper component of interval. Must be greater than upper.
        expand_factor: How much to (multiplicatively) increase the adjustment by on
            each iteration. The magnitude of the adjustment of the necessary bound is
            given by ``(init_upper - init_lower)*expand_factor**iteration``. Defaults to
            2.0.
    """
    fn_lower, fn_upper = func(lower), func(upper)

    class _State(NamedTuple):
        lower: Real[Array, ""]
        upper: Real[Array, ""]
        expand_by: Real[Array, ""]
        lower_fn_sign: Real[Array, ""]
        upper_fn_sign: Real[Array, ""]
        iteration: Int[Array, ""] = jnp.array(0)

    def cond_fn(state):
        return state.lower_fn_sign == state.upper_fn_sign

    def body_fn(state):
        sign = state.lower_fn_sign  # Note we know the signs match from cond_fn
        lower_update = jnp.where(sign == 1, state.lower - state.expand_by, state.upper)
        upper_update = jnp.where(sign == 1, state.lower, state.upper + state.expand_by)
        return _State(
            lower=lower_update,
            upper=upper_update,
            expand_by=state.expand_by * expand_factor,
            lower_fn_sign=jnp.sign(func(lower_update)),
            upper_fn_sign=jnp.sign(func(upper_update)),
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

    # In case we land on root exactly, return lower==upper==root
    lower, upper = state.lower, state.upper
    lower = jnp.where(state.upper_fn_sign == 0, upper, lower)
    upper = jnp.where(state.lower_fn_sign == 0, lower, upper)
    return lower, upper, state.iteration
