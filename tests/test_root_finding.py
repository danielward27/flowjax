from functools import partial

import jax
import jax.numpy as jnp
import pytest

from flowjax.root_finding import (
    WhileResult,
    _adapt_interval_to_include_root,
    bisect_check_expand_search,
    bisection_search,
    elementwise_autoregressive_bisection,
)


def target_function(x):
    return x + 4


def test_adapt_interval_to_include_root():
    adapt_result = _adapt_interval_to_include_root(
        target_function,
        lower=jnp.array(1.1),
        upper=jnp.array(1.2),
    )

    assert adapt_result.state.lower < -4
    assert adapt_result.state.upper < 1.1  # Upper should improve too

    # If already includes root, shouldn't change anything
    init_lower, init_upper = jnp.array(-10), jnp.array(10)
    adapt_result = _adapt_interval_to_include_root(
        target_function,
        lower=init_lower,
        upper=init_upper,
    )
    assert adapt_result.state.lower == init_lower
    assert adapt_result.state.upper == init_upper
    assert adapt_result.steps == 0


true_root = -4
adapt_exact_test_cases = [
    (true_root, 10, 0),
    (-10, true_root, 0),
    (-2, 0, 1),  # Lower adapted and hits root
    (-8, -6, 1),  # Upper adapted and hits root
]


@pytest.mark.parametrize(
    ("lower", "upper", "expected_steps"),
    adapt_exact_test_cases,
)
def test_adapt_interval_to_include_root_exact(lower, upper, expected_steps):
    # Tests cases where the exact root is found
    adapt_result = _adapt_interval_to_include_root(
        target_function,
        lower=jnp.array(lower),
        upper=jnp.array(upper),
    )
    assert adapt_result.state.lower == true_root
    assert adapt_result.state.upper == true_root
    assert adapt_result.steps == expected_steps


def test_bisection_search():
    max_steps = 200

    root, (adapt_state, bisect_state) = bisection_search(
        target_function,
        lower=jnp.array(-10),
        upper=jnp.array(10),
        max_steps=max_steps,
    )

    assert root == pytest.approx(-4, abs=1e-5)
    assert bisect_state.steps < max_steps
    assert adapt_state.steps == 0

    # Check max_steps terminates loop
    root, aux = bisection_search(
        target_function,
        lower=jnp.array(-10),
        upper=jnp.array(10),
        max_steps=1,
        throw=False,
    )
    assert aux[1].steps == 1

    # Check can adapt interval if needed
    root, (adapt_state, _) = bisection_search(
        target_function,
        lower=jnp.array(3),
        upper=jnp.array(4),
        max_steps=200,
    )
    assert root == pytest.approx(-4, abs=1e-5)
    assert adapt_state.steps > 0

    root, (adapt_state, _) = bisection_search(
        target_function,
        lower=-10,
        upper=-9,
        max_steps=200,
    )


def test_bisection_search_exact():
    # Tests cases where the exact root is found
    root, aux = bisection_search(
        target_function,
        lower=jnp.array(true_root - 2),
        upper=jnp.array(true_root + 2),
        max_steps=200,
    )
    assert root == true_root
    assert aux[1].steps == 1


autoregressive_test_cases = {
    "elementwise": partial(
        elementwise_autoregressive_bisection,
        lower=jnp.full((3,), -10),
        upper=jnp.full((3,), 10),
    ),
    "elementwise high init": partial(
        elementwise_autoregressive_bisection,
        lower=jnp.full((3,), 10),
        upper=jnp.full((3,), 11),
    ),
    "elementwise low init": partial(
        elementwise_autoregressive_bisection,
        lower=jnp.full((3,), -11),
        upper=jnp.full((3,), -10),
    ),
    "elementwise exact init lower": partial(
        elementwise_autoregressive_bisection,
        lower=jnp.array([0, -1, -1]),
        upper=jnp.full((3,), 10),
    ),
    "elementwise exact init upper": partial(
        elementwise_autoregressive_bisection,
        lower=jnp.full((3,), -10),
        upper=jnp.array([0, -1, -1]),
    ),
    "bisect_check_expand_search": partial(
        bisect_check_expand_search,
        midpoint=jnp.zeros(3),
        width=5,
    ),
    "bisect_check_expand_search high init": partial(
        bisect_check_expand_search,
        midpoint=jnp.full((3,), 10, float),
        width=0.1,
    ),
    "bisect_check_expand_search low init": partial(
        bisect_check_expand_search,
        midpoint=jnp.full((3,), -10, float),
        width=0.1,
    ),
    "bisect_check_expand_search exact init": partial(
        bisect_check_expand_search,
        midpoint=jnp.array([0, -1, -1], float),
        width=5,
    ),
}


def autoregressive_func(array):
    return jnp.cumsum(array) + jnp.arange(3)


@pytest.mark.parametrize(
    "root_finder",
    autoregressive_test_cases.values(),
    ids=autoregressive_test_cases.keys(),
)
def test_autoregressive_root_finders(root_finder):
    root, aux = root_finder(fn=autoregressive_func, max_steps=1000)
    assert root == pytest.approx(jnp.array([0, -1, -1]), abs=1e-4)

    def map_fn(leaf):
        if isinstance(leaf, WhileResult):
            assert ~jnp.any(leaf.reached_max_steps)

    jax.tree_util.tree_map(
        map_fn,
        aux,
        is_leaf=lambda leaf: isinstance(leaf, WhileResult),
    )
