import jax.numpy as jnp
import pytest

from flowjax.bijections import TriangularAffine


def test_TriangularAffine():
    "Test mainly to check min_diag does not cause unexpected results on initialisation."
    loc = jnp.array([1, 2])
    lower = jnp.array([[2, 0], [0.5, 3]])

    bijection = TriangularAffine(loc, lower)
    x = jnp.ones(2)
    y = bijection.transform(x)
    expected = lower @ x + loc
    assert y == pytest.approx(expected)
