from flowjax.bijections.parameterised import Affine
import jax.numpy as jnp
import pytest


def test_Affine():
    d = 5
    b = Affine()
    x = jnp.arange(5)
    loc, scale = jnp.ones(d), jnp.ones(d)
    y = b.transform(x, loc, scale)
    x_reconstructed = b.inverse(y, loc, scale)
    assert x == pytest.approx(x_reconstructed)
    assert (x != y).all()
