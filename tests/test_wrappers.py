import jax
import jax.numpy as jnp
import pytest

from flowjax.bijections import Exp
from flowjax.wrappers import BijectionReparam, unwrap


def test_BijectionReparam():

    with pytest.raises(jax.lib.xla_extension.XlaRuntimeError, match="Exp"):
        BijectionReparam(-jnp.ones(3), Exp())

    param = jnp.array([jnp.inf, 1, 2])
    wrapped = BijectionReparam(param, Exp())
    assert pytest.approx(unwrap(wrapped)) == param
    assert pytest.approx(wrapped.arr) == jnp.log(param)
