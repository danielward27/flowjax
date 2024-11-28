""""Bisection search algorithm.

This is useful for inverting some bijections that do not have a known inverse.
These methods should be considered experimental and may be changed without warning.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Int, Real, ScalarLike


def root_finder_to_inverter(
    root_finder: Callable[[Callable], tuple[Array, Any]],
):
    """Convert root finder to an "inverter".

    For use with :class:`~flowjax.bijections.NumericalInverse`. Root finder should have
    arguments provided e.g. using ``functools.partial``. We assume the root finder
    takes a single argument, the function to invert, and returns a tuple, with the root
    as the first element and any auxillary information as the second.
    """

    @eqx.filter_jit
    def inverter(bijection, y, condition=None):

        def fn(x):
            return bijection.transform(x, condition) - y

        return root_finder(fn)[0]

    return inverter


class _BisectionState(eqx.Module):
    lower: Float[Array, " "]
    upper: Float[Array, " "]


class _AdaptIntervalState(eqx.Module):
    lower: Real[Array, ""]
    upper: Real[Array, ""]
    expand_by: Real[Array, ""]
    fn_lower: Real[Array, ""]
    fn_upper: Real[Array, ""]


def _adapt_interval_to_include_root(
    fn,
    *,
    lower: Real[Array, ""],
    upper: Real[Array, ""],
    expand_factor: float = 2.0,
    max_iter: int = 1000,
    error: bool = True,
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
    fn_lower, fn_upper = fn(lower), fn(upper)

    def handle_exact_root(state: _AdaptIntervalState):
        either_zero = jnp.logical_or(state.fn_lower == 0, state.fn_upper == 0)
        maybe_root = jnp.where(state.fn_lower == 0, state.lower, state.upper)
        return _AdaptIntervalState(
            lower=jnp.where(either_zero, maybe_root, state.lower),
            upper=jnp.where(either_zero, maybe_root, state.upper),
            expand_by=state.expand_by,
            fn_lower=jnp.where(either_zero, 0, state.fn_lower),
            fn_upper=jnp.where(either_zero, 0, state.fn_upper),
        )

    def cond_fn(state):
        signs_match = jnp.sign(state.fn_lower) == jnp.sign(state.fn_upper)
        not_zero = jnp.logical_and(state.fn_lower != 0, state.fn_upper != 0)
        return jnp.logical_and(signs_match, not_zero)

    def body_fn(state):
        is_positive = state.fn_lower > 0  # Note we know the signs match from cond_fn
        lower = jnp.where(is_positive, state.lower - state.expand_by, state.upper)
        upper = jnp.where(is_positive, state.lower, state.upper + state.expand_by)
        eval_at = jnp.where(is_positive, lower, upper)
        f_eval = fn(eval_at)
        fn_lower = jnp.where(is_positive, f_eval, state.fn_upper)
        fn_upper = jnp.where(is_positive, state.fn_lower, f_eval)
        return _AdaptIntervalState(
            lower=lower,
            upper=upper,
            expand_by=state.expand_by * expand_factor,
            fn_lower=fn_lower,
            fn_upper=fn_upper,
        )

    lower, upper = jnp.asarray(lower, float), jnp.asarray(upper, float)

    init_state = _AdaptIntervalState(
        lower=lower,
        upper=upper,
        expand_by=upper - lower,
        fn_lower=fn_lower,
        fn_upper=fn_upper,
    )
    init_state = handle_exact_root(init_state)
    result = _max_iter_while(
        cond_fn,
        body_fn,
        init_state,
        max_iter=max_iter,
        error=error,
    )
    return eqx.tree_at(lambda res: res.state, result, handle_exact_root(result.state))


def elementwise_autoregressive_bisection_search(
    fn: Callable,
    *,
    lower: Array,
    upper: Array,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    max_iter: int = 1000,
    error: bool = True,
) -> tuple[Array, tuple[_AdaptIntervalState, _BisectionState]]:
    """Bisection search for a monotonic increasing autoregressive function.

    We scan over the inputs finding the root element by element, assuming that
    each input only depends on previous inputs in the input array. This is useful
    for inverting some bijections without a known inverse, such as those used in block
    neural autoregressive flows. Note that tol refers to the tolerance used in each
    run of the bisection search. This means the found solution may not necessarily
    be within the tolerance for all elements, as errors can accumulate in each step.

    Args:
        fn: The monotonically increasing autoregressive function.
        tol: Tolerance of a solution. Note due to accumulation of errors at each step,
            the found solution may not fall within the given tolerance.
        lower: Lower bound of the initial interval where the root is expected. Lower and
            upper should broadcast to the dimensionality of the root finding problem.
        upper: Upper bound of the initial interval where the root is expected.
        max_iter: Maximum number of iterations for each element. Defaults to 1000.

    """
    lower, upper = jnp.broadcast_arrays(lower, upper)

    def scan_fn(init, xs):
        y, idx = init
        lower, upper = xs

        def scalar_fn(x):
            x = y.at[idx].set(x)
            return fn(x)[idx]

        root, aux = bisection_search(
            scalar_fn,
            rtol=rtol,
            atol=atol,
            lower=lower,
            upper=upper,
            max_iter=max_iter,
            error=error,
        )
        return (y.at[idx].set(root), idx + 1), aux

    init = ((upper + lower) / 2, 0)
    (root, _), aux = lax.scan(scan_fn, init=init, xs=(lower, upper), length=len(lower))
    return root, aux


def bisection_search(
    func: Callable,
    *,
    lower: ScalarLike,
    upper: ScalarLike,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    max_iter: int = 1000,
    error: bool = True,
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
        max_iter: Maximum number of iterations. Used independently for both adapting the
            interval bounds, and for the bisection search. Defaults to 1000.
        error_on_max_iter: bool = True,

    """
    if max_iter < 0:
        raise ValueError("max_iter must be positive.")
    if atol <= 0:
        raise ValueError("tol must be positive.")

    adapt_result = _adapt_interval_to_include_root(
        func,
        lower=jnp.asarray(lower, float),
        upper=jnp.asarray(upper, float),
        max_iter=max_iter,
        error=error,
    )

    def cond_fn(state):
        return ~jnp.isclose((state.upper - state.lower) / 2, 0, rtol=rtol, atol=atol)

    def body_fn(state):
        midpoint = (state.lower + state.upper) / 2
        f_midpoint = func(midpoint)
        lower = jnp.where(f_midpoint > 0, state.lower, midpoint)
        upper = jnp.where(f_midpoint > 0, midpoint, state.upper)
        lower = jnp.where(f_midpoint == 0, midpoint, lower)
        upper = jnp.where(f_midpoint == 0, midpoint, upper)
        return _BisectionState(lower, upper)

    init_state = _BisectionState(adapt_result.state.lower, adapt_result.state.upper)

    bisect_result = _max_iter_while(
        cond_fn,
        body_fn,
        init_state,
        max_iter=max_iter,
        error=error,
    )
    root = (bisect_result.state.lower + bisect_result.state.upper) / 2
    return root, (adapt_result, bisect_result)


def _max_iter_while(cond_fun, body_fun, init_val, *, max_iter: int, error: bool = True):
    """While loop with max iterations.

    Returns:
        A dictionary with keys "state" and "iterations".
    """

    def cond_fn_max_iter(state):
        return jnp.logical_and(cond_fun(state[0]), state[1] < max_iter)

    def body_fn_with_iter(state):
        return body_fun(state[0]), state[1] + 1

    state, iterations = lax.while_loop(
        cond_fn_max_iter,
        body_fn_with_iter,
        (init_val, jnp.array(0)),
    )
    if error:
        state = eqx.error_if(
            state,
            iterations == max_iter,
            "Maximum iterations reached in while loop. Consider increasing max_iter.",
        )

    class _Result(eqx.Module):
        state: Any
        iterations: Int[Array, " "]

    return _Result(state, iterations)
