from realnvp.bijections.permute import Permute
import jax.numpy as jnp
import pytest


def test_Permute():
    x = jnp.arange(4)
    permute = Permute(jnp.array([3, 2, 1, 0]))
    y = permute.transform(x)
    x_reconstructed = permute.inverse(y)
    assert x == pytest.approx(x_reconstructed)
    assert (x != y).sum()
