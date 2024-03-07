import jax
import jax.numpy as jnp
import pytest

from flowjax.bijections import Exp
from flowjax.wrappers import BijectionReparam, Diagonal, unwrap


def test_BijectionReparam():

    with pytest.raises(jax.lib.xla_extension.XlaRuntimeError, match="Exp"):
        BijectionReparam(-jnp.ones(3), Exp())

    param = jnp.array([jnp.inf, 1, 2])
    wrapped = BijectionReparam(param, Exp())
    assert pytest.approx(unwrap(wrapped)) == param
    assert pytest.approx(wrapped.arr) == jnp.log(param)


def test_Diagonal():

    diag = Diagonal(jnp.ones((2, 2)))  # This should unwrap to 3D unlike jnp.diag
    assert unwrap(diag).shape == (2, 2, 2)

    expected = jnp.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    assert pytest.approx(unwrap(diag)) == expected
