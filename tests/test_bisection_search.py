import jax.numpy as jnp
import pytest

from flowjax.bisection_search import (
    AutoregressiveBisectionInverter,
    _adapt_interval_to_include_root,
    _autoregressive_bisection_search,
    _bisection_search,
)


def target_function(x):
    return x + 4


def test_adapt_interval_to_include_root():
    lower, upper, _ = _adapt_interval_to_include_root(
        target_function,
        lower=jnp.array(1.1),
        upper=jnp.array(1.2),
    )

    assert lower < -4
    assert upper < 1.1  # Upper should improve too

    # If already includes root, shouldn't change anything
    init_lower, init_upper = jnp.array(-10), jnp.array(10)
    lower, upper, iterations = _adapt_interval_to_include_root(
        target_function,
        lower=init_lower,
        upper=init_upper,
    )
    assert lower == init_lower
    assert upper == init_upper
    assert iterations == 0


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
    lower, upper, iterations = _adapt_interval_to_include_root(
        target_function,
        lower=jnp.array(lower),
        upper=jnp.array(upper),
    )
    assert lower == true_root
    assert upper == true_root
    assert iterations == expected_iterations


def test_bisection_search():
    tol = 0.1
    max_iter = 200

    root, adapt_iterations, iterations = _bisection_search(
        target_function,
        lower=jnp.array(-10),
        upper=jnp.array(10),
        tol=tol,
        max_iter=max_iter,
    )

    assert root == pytest.approx(-4, abs=tol)
    assert iterations < max_iter
    assert adapt_iterations == 0

    # Check max_iter terminates loop
    root, adapt_iterations, iterations = _bisection_search(
        target_function,
        lower=jnp.array(-10),
        upper=jnp.array(10),
        tol=tol,
        max_iter=0,
    )
    assert iterations == 0

    # Check can adapt interval if needed
    root, adapt_iterations, iterations = _bisection_search(
        target_function,
        lower=jnp.array(3),
        upper=jnp.array(4),
        tol=tol,
        max_iter=200,
    )
    assert root == pytest.approx(-4, abs=tol)
    assert adapt_iterations > 0


def test_bisection_search_exact():
    # Tests cases where the exact root is found
    root, _, iterations = _bisection_search(
        target_function,
        lower=jnp.array(true_root - 2),
        upper=jnp.array(true_root + 2),
        tol=0.1,
        max_iter=200,
    )
    assert root == true_root
    assert iterations == 1


def test_autoregressive_bisection_search():
    def autoregressive_func(array):
        return jnp.cumsum(array) + jnp.arange(3)

    tol = 1e-5
    result = _autoregressive_bisection_search(
        autoregressive_fn=autoregressive_func,
        tol=tol,
        lower=jnp.array(-10),
        upper=jnp.array(10),
        length=3,
        max_iter=200,
    )
    assert result == pytest.approx(jnp.array([0, -1, -1]), abs=tol * 3)


def test_autoregressive_bijection_bisection_inverter():
    inverter = AutoregressiveBisectionInverter()

    from flowjax.bijections import Affine

    affine = Affine(jnp.arange(3), 0.2)
    y = jnp.arange(3)
    x_bisection = inverter(affine, y)
    x_true = affine.inverse(y)

    # Note due to accumulation of errors we cannot use tol below.
    assert x_bisection == pytest.approx(x_true, abs=inverter.tol * 3)
