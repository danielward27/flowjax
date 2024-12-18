"""Root finding methods.

Root finding methods can be useful for inverting bijections without an analytical
inverse. This module should be considered experimental and may be changed without
warning  - thus far little published research exists for root finding algorithms
specialised for autoregressive increasing functions. 
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PyTree, Real, ScalarLike

from flowjax.utils import arraylike_to_array


class WhileResult(eqx.Module):
    """Result of max_steps_while loop, with state, steps and reached_max_steps."""

    state: Any
    steps: Int[Array, " "]
    reached_max_steps: Bool[Array, " "]


def max_steps_while_loop(
    cond_fun: Callable,
    body_fun: Callable,
    init_val: PyTree,
    *,
    max_steps: int,
    throw: bool = True,
    error_context: str = "",
):
    """While loop with max steps, returning ``WhileResult``.

    Args:
        cond_fun: Function of type ``a -> Bool``.
        body_fun: Function of type ``a -> a``.
        init_val: Initial value of type ``a``.
        max_steps: Maximum number of steps.
        throw: Whether to error if ``max_steps`` is reached. Defaults to True.
        error_context: Additional context for the error if ``max_steps`` is reached.
            Defaults to "".
    """

    def cond_fn_max_steps(state):
        return jnp.logical_and(cond_fun(state[0]), state[1] < max_steps)

    def body_fn_with_iter(state):
        return body_fun(state[0]), state[1] + 1

    state, steps = lax.while_loop(
        cond_fn_max_steps,
        body_fn_with_iter,
        (init_val, jnp.array(0)),
    )
    result = WhileResult(state, steps, steps == max_steps)
    if throw:
        result = eqx.error_if(
            result,
            result.reached_max_steps,
            "Maximum steps reached in while loop. Consider increasing max_steps."
            + error_context,
        )
    return result


def root_finder_to_inverter(
    root_finder: Callable[[Callable], tuple[Array, Any]],
):
    """Utility to convert root finder to an "inverter".

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


class BisectCheckExpandSearchState(eqx.Module, strict=True):
    """State for bisect_check_expand_search."""

    midpoint: Float[Array, " d"]
    width: Float[Array, " d"]
    confidence: Float[Array, " d"]
    last_fn_sign: Float[Array, " d"]
    f_midpoint: Float[Array, " d"]


@eqx.filter_jit
def bisect_check_expand_search(
    fn: Callable,
    midpoint: ArrayLike,
    width: ArrayLike = 2,
    *,
    atol: float = 1e-5,
    max_steps: int = 5000,
    throw: bool = True,
    max_width: float | int | None = None,
) -> tuple[Array, WhileResult]:
    """Bisect check expand search for an increasing autoregressive function.

    This algorithm uses elementwise bisection-like updates whilst tracking sign changes
    in fn across iterations as measure of reliability of the bracketing interval. If the
    same sign is repeatedly observed, this may indicate the bracketing interval is
    unreliable, in which case the corresponding bound is checked and expanded if needed.
    This is a greedy approach that allows updating all elements of the root
    simultaneously, which tends to be much faster than an elementwise bisection search.

    Args:
        fn: Autoregressive monotonically increasing function.
        midpoint: The initial midpoint of the bracketing interval. Midpoint and width
            should broadcast to the dimensionality of fn.
        width: The width used to define the interval [midpoint-width, midpoint+width].
            Midpoint and width should broadcast to the dimensionality of fn.
            Defaults to 2.
        atol: Absolute tolerance level. Defaults to 1e-5.
        max_steps: Maximum number of steps to use. Defaults to 5000.
        throw: Whether to error if the max_steps is reached. Defaults to True.
        max_width: The maximum width of the bracketing interval. Defaults to
            ``10*width``.


    Returns:
        A tuple, with the array root and a ``WhileResult`` of auxilary information.
    """
    max_width = 10 * width if max_width is None else max_width
    midpoint = arraylike_to_array(midpoint, dtype=float)
    width = arraylike_to_array(width, dtype=float)
    midpoint, width = jnp.broadcast_arrays(midpoint, width)

    if atol <= 0:
        raise ValueError("tol must be positive.")

    width = eqx.error_if(width, width <= 0, "width must be positive.")
    f_midpoint = fn(midpoint)

    init_state = BisectCheckExpandSearchState(
        midpoint=midpoint - 0.5 * jnp.sign(f_midpoint) * width,
        width=width,
        confidence=jnp.ones_like(width),
        last_fn_sign=jnp.sign(f_midpoint),
        f_midpoint=jnp.full_like(midpoint, jnp.inf),
    )

    def cond_fn(state: BisectCheckExpandSearchState):
        return ~jnp.isclose(jnp.mean(jnp.abs(state.f_midpoint)), 0, rtol=0, atol=atol)

    def body_fn(state: BisectCheckExpandSearchState):
        f_midpoint = fn(state.midpoint)
        signs_changed = jnp.sign(f_midpoint) != state.last_fn_sign
        confidence = jnp.where(signs_changed, 1, state.confidence / 2)

        scale = 2 ** (1 - jnp.minimum(4 * confidence, 2))  # .5, .5, 1, ~1.4 ...  2
        width = jnp.minimum(scale * state.width, max_width)
        midpoint = state.midpoint - jnp.sign(f_midpoint) * width

        return BisectCheckExpandSearchState(
            midpoint=midpoint,
            width=width,
            last_fn_sign=jnp.sign(f_midpoint),
            confidence=confidence,
            f_midpoint=f_midpoint,
        )

    result = max_steps_while_loop(
        cond_fn,
        body_fn,
        init_state,
        max_steps=max_steps,
        throw=throw,
        error_context="If increasing ``max_steps`` does not help, consider checking "
        "the problem is well formulated, and using instead "
        "``elementwise_autoregressive_bisection``, which will be slow to converge but "
        "possibly more robust.",
    )
    return result.state.midpoint, result


class BisectionState(eqx.Module):
    """State for a bisection search."""

    lower: Float[Array, " "]
    upper: Float[Array, " "]


class _AdaptIntervalState(eqx.Module):
    """State for adapting an interval to include a root."""

    lower: Real[Array, ""]
    upper: Real[Array, ""]
    expand_by: Real[Array, ""]
    fn_lower: Real[Array, ""]
    fn_upper: Real[Array, ""]


def bisection_search(
    func: Callable,
    lower: ScalarLike,
    upper: ScalarLike,
    *,
    atol: float = 1e-5,
    max_steps: int = 1000,
    throw: bool = True,
) -> tuple[Array, tuple[WhileResult, WhileResult]]:
    """Bisection search algorithm to find a root of a scalar increasing function.

    Note that the implementation assumes that the function is monotonically increasing.
    If the initial limits do not include the root then the interval is adapted using
    ``expand_interval_to_include_root``.

    Args:
        func: Scalar increasing function to find the root for.
        lower: Lower bound of the initial interval where the root is expected.
        upper: Upper bound of the initial interval where the root is expected.
        atol: Absolute tolerance level. Defaults to 1e-5.
        max_steps: Maximum number of steps to use, passed to both the adaptation of
            the intervals, and the bisection search. Defaults to 1000.
        throw: Whether to error if the maximum numer of steps is reached in the
            bisection search. Note if the adapation of the interval fails, an error will
            be raised regardless of the value of error. Defaults to True.
    """
    if max_steps < 0:
        raise ValueError("max_steps must be positive.")
    if atol <= 0:
        raise ValueError("atol must be positive.")

    adapt_result = _adapt_interval_to_include_root(
        func,
        lower=jnp.asarray(lower, float),
        upper=jnp.asarray(upper, float),
        max_steps=max_steps,
        throw=True,  # Probably safer to error if root not spanned
    )

    def cond_fn(state):
        return ~jnp.isclose((state.upper - state.lower) / 2, 0, atol=atol)

    def body_fn(state):
        midpoint = (state.lower + state.upper) / 2
        f_midpoint = func(midpoint)
        lower = jnp.where(f_midpoint > 0, state.lower, midpoint)
        upper = jnp.where(f_midpoint > 0, midpoint, state.upper)
        lower = jnp.where(f_midpoint == 0, midpoint, lower)
        upper = jnp.where(f_midpoint == 0, midpoint, upper)
        return BisectionState(lower, upper)

    init_state = BisectionState(adapt_result.state.lower, adapt_result.state.upper)

    bisect_result = max_steps_while_loop(
        cond_fn,
        body_fn,
        init_state,
        max_steps=max_steps,
        throw=throw,
    )
    root = (bisect_result.state.lower + bisect_result.state.upper) / 2
    return root, (adapt_result, bisect_result)


def _adapt_interval_to_include_root(
    fn,
    lower: Real[Array, ""],
    upper: Real[Array, ""],
    *,
    expand_factor: float | int = 2,
    max_steps: int = 1000,
    throw: bool = True,
):
    """Adjust the interval to include the root of an increasing function.

    Args:
        fn: A scalar increasing function.
        lower: Lower component of interval. Must be less than upper.
        upper: Upper component of interval. Must be greater than upper.
        expand_factor: How much to (multiplicatively) increase the adjustment by on
            each step. The magnitude of the adjustment of the necessary bound is
            given by ``(init_upper - init_lower)*expand_factor**step``. Defaults to
            2.0.
        max_steps: Maximum number of steps to use. Defaults to 1000.
        throw: Whether to error if the maximum numer of steps is reached. Defaults
            to True.
    """
    lower = eqx.error_if(lower, lower >= upper, "Lower must be less than upper.")
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

    lower = arraylike_to_array(lower, dtype=float)
    upper = arraylike_to_array(upper, dtype=float)

    init_state = _AdaptIntervalState(
        lower=lower,
        upper=upper,
        expand_by=upper - lower,
        fn_lower=fn_lower,
        fn_upper=fn_upper,
    )
    init_state = handle_exact_root(init_state)
    result = max_steps_while_loop(
        cond_fn,
        body_fn,
        init_state,
        max_steps=max_steps,
        throw=throw,
    )
    return eqx.tree_at(lambda res: res.state, result, handle_exact_root(result.state))


def elementwise_autoregressive_bisection(
    fn: Callable,
    lower: Array,
    upper: Array,
    *,
    atol: float = 1e-5,
    max_steps: int = 1000,
    throw: bool = True,
) -> tuple[Array, tuple[WhileResult, WhileResult]]:
    """Bisection search for a monotonic increasing autoregressive function.

    We scan over the inputs finding the root element by element, assuming that
    each input only depends on previous inputs in the input array. If the root is not
    included in the initial interval, it is iteratively expanded. Note the full
    ``fn`` is evaluated at each iteration, but only a single element of the approximate
    multivariate root is updated, making this approach computationally costly in
    higher dimensional problems.

    Args:
        fn: The monotonically increasing autoregressive function.
        lower: Lower bound of the initial interval where the root is expected. Lower and
            upper should broadcast to the dimensionality of the root finding problem.
        upper: Upper bound of the initial interval where the root is expected.
        atol: Absolute tolerance level. Defaults to 1e-5.
        max_steps: Maximum number of steps for each element. Defaults to 1000.
        throw: Whether to error if the maximum numer of steps is reached. Defaults
            to True.

    Returns:
        The root, and a tuple containing the auxilary information for expanding the
            interval, and the bisection search.
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
            atol=atol,
            lower=lower,
            upper=upper,
            max_steps=max_steps,
            throw=throw,
        )
        return (y.at[idx].set(root), idx + 1), aux

    init = ((upper + lower) / 2, 0)
    (root, _), aux = lax.scan(scan_fn, init=init, xs=(lower, upper), length=len(lower))
    return root, aux
