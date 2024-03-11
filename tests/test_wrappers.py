import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from flowjax.bijections import Exp
from flowjax.wrappers import BijectionReparam, Lambda, unwrap


def test_BijectionReparam():

    with pytest.raises(jax.lib.xla_extension.XlaRuntimeError, match="Exp"):
        BijectionReparam(-jnp.ones(3), Exp())

    param = jnp.array([jnp.inf, 1, 2])
    wrapped = BijectionReparam(param, Exp())
    assert pytest.approx(unwrap(wrapped)) == param
    assert pytest.approx(wrapped.arr) == jnp.log(param)


def test_Lambda():
    diag = Lambda(jnp.diag, jnp.ones(3))
    assert pytest.approx(jnp.eye(3)) == unwrap(diag)

    # Test works when vmapped (note diag does not follow standard vectorization rules)
    v_diag = eqx.filter_vmap(Lambda)(jnp.diag, jnp.ones((4, 3)))
    expected = eqx.filter_vmap(jnp.eye, axis_size=4)(3)
    assert pytest.approx(expected) == unwrap(v_diag)
