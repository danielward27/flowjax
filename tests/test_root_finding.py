import jax.numpy as jnp
import pytest

from flowjax.root_finding import (
    _adapt_interval_to_include_root,
    _AdaptIntervalState,
    _BisectionState,
    bisection_search,
    elementwise_autoregressive_bisection_search,
)


def target_function(x):
    return x + 4


def test_adapt_interval_to_include_root():
    adapt_result = _adapt_interval_to_include_root(
        target_function,
        lower=jnp.array(1.1),
        upper=jnp.array(1.2),
    )

    assert adapt_result["state"].lower < -4
    assert adapt_result["state"].upper < 1.1  # Upper should improve too

    # If already includes root, shouldn't change anything
    init_lower, init_upper = jnp.array(-10), jnp.array(10)
    adapt_result["state"] = _adapt_interval_to_include_root(
        target_function,
        lower=init_lower,
        upper=init_upper,
    )
    assert adapt_result["state"].lower == init_lower
    assert adapt_result["state"].upper == init_upper
    assert adapt_result["iterations"] == 0


true_root = -4
adapt_exact_test_cases = [
    (true_root, 10, 0),
    (-10, true_root, 0),
    (-2, 0, 1),  # Lower adapted and hits root
    (-8, -6, 1),  # Upper adapted and hits root
]


@pytest.mark.parametrize(
    ("lower", "upper", "expected_iterations"),
    adapt_exact_test_cases,
)
def test_adapt_interval_to_include_root_exact(lower, upper, expected_iterations):
    # Tests cases where the exact root is found
    adapt_result = _adapt_interval_to_include_root(
        target_function,
        lower=jnp.array(lower),
        upper=jnp.array(upper),
    )
    assert adapt_result.state.lower == true_root
    assert adapt_result.state.upper == true_root
    assert adapt_result.iterations == expected_iterations


def test_bisection_search():
    max_iter = 200

    root, (adapt_state, bisect_state) = bisection_search(
        target_function,
        lower=jnp.array(-10),
        upper=jnp.array(10),
        max_iter=max_iter,
    )

    assert root == pytest.approx(-4, abs=1e-5)
    assert bisect_state.iterations < max_iter
    assert adapt_state.iterations == 0

    # Check max_iter terminates loop
    root, aux = bisection_search(
        target_function,
        lower=jnp.array(-10),
        upper=jnp.array(10),
        max_iter=0,
        error=False,
    )
    assert aux[1].iterations == 0

    # Check can adapt interval if needed
    root, (adapt_state, _) = bisection_search(
        target_function,
        lower=jnp.array(3),
        upper=jnp.array(4),
        max_iter=200,
    )
    assert root == pytest.approx(-4, abs=1e-5)
    assert adapt_state.iterations > 0

    root, (adapt_state, _) = bisection_search(
        target_function,
        lower=-10,
        upper=-9,
        max_iter=200,
    )


def test_bisection_search_exact():
    # Tests cases where the exact root is found
    root, aux = bisection_search(
        target_function,
        lower=jnp.array(true_root - 2),
        upper=jnp.array(true_root + 2),
        max_iter=200,
    )
    assert root == true_root
    assert aux[1].iterations == 1


def test_autoregressive_bisection_search():
    def autoregressive_func(array):
        return jnp.cumsum(array) + jnp.arange(3)

    root, aux = elementwise_autoregressive_bisection_search(
        fn=autoregressive_func,
        lower=jnp.full((3,), -10),
        upper=jnp.full((3,), 10),
        max_iter=200,
    )
    assert root == pytest.approx(jnp.array([0, -1, -1]), abs=1e-4)
    assert isinstance(aux[0], _AdaptIntervalState)
    assert isinstance(aux[1], _BisectionState)
    assert aux[0].lower.shape == (3,)
    assert aux[1].lower.shape == (3,)
