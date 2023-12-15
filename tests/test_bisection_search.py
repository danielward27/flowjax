import jax.numpy as jnp
import pytest

from flowjax.bisection_search import (
    AutoregressiveBisectionInverter,
    adapt_interval_to_include_root,
    autoregressive_bisection_search,
    bisection_search,
)


def target_function(x):
    return x + 4


def test_adapt_interval_to_include_root():
    lower, upper, _ = adapt_interval_to_include_root(
        target_function,
        lower=1.1,
        upper=1.2,
    )

    assert lower < -4
    assert upper < 1.1  # Upper should improve too

    # If already includes root, shouldn't change anything
    init_lower, init_upper = -10, 10
    lower, upper, iterations = adapt_interval_to_include_root(
        target_function,
        lower=init_lower,
        upper=init_upper,
    )
    assert lower == init_lower
    assert upper == init_upper
    assert iterations == 0


def test_bisection_search():
    tol = 0.1
    max_iter = 200

    root, adapt_iterations, iterations = bisection_search(
        target_function,
        lower=-10,
        upper=10,
        tol=tol,
        max_iter=max_iter,
    )

    assert root == pytest.approx(-4, abs=tol)
    assert iterations < max_iter
    assert adapt_iterations == 0

    # Check max_iter terminates loop
    root, adapt_iterations, iterations = bisection_search(
        target_function,
        lower=-10,
        upper=10,
        tol=tol,
        max_iter=0,
    )
    assert iterations == 0

    # Check can adapt interval if needed
    root, adapt_iterations, iterations = bisection_search(
        target_function,
        lower=3,
        upper=4,
        tol=tol,
        max_iter=200,
    )
    assert root == pytest.approx(-4, abs=tol)
    assert adapt_iterations > 0


def test_autoregressive_bisection_search():
    def autoregressive_func(array):
        return jnp.cumsum(array) + jnp.arange(3)

    tol = 1e-5
    result = autoregressive_bisection_search(
        autoregressive_fn=autoregressive_func,
        tol=tol,
        lower=-10,
        upper=10,
        length=3,
        max_iter=200,
    )
    assert result == pytest.approx(jnp.array([0, -1, -1]), abs=tol)


def test_autoregressive_bijection_bisection_inverter():
    inverter = AutoregressiveBisectionInverter()

    from flowjax.bijections import Affine

    affine = Affine(jnp.arange(3), 0.2)
    y = jnp.arange(3)
    x_bisection = inverter(affine, y)
    x_true = affine.inverse(y)

    # Note due to accumulation of errors we cannot use tol below.
    assert x_bisection == pytest.approx(x_true, abs=inverter.tol * 3)
