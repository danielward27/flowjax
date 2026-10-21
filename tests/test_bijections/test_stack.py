import math

import equinox as eqx
import jax.numpy as jnp
import pytest

from flowjax.bijections import Affine, Stack


@pytest.mark.parametrize(
    ("shape", "axis"),
    [((3,), -1), ((3, 4), -1), ((3, 4), -2), ((3, 4), -3), ((), -1)],
)
def test_stack_negative_axis_matches_array_stack(shape, axis):
    bijection = Stack(
        [Affine(jnp.ones(shape), 2), Affine(-jnp.ones(shape), 3)], axis=axis
    )
    x = jnp.stack([jnp.zeros(shape), jnp.ones(shape)], axis=axis)
    expected = jnp.stack([jnp.ones(shape), jnp.full(shape, 2.0)], axis=axis)
    expected_log_det = math.prod(shape) * jnp.log(6.0)

    assert bijection.shape == x.shape
    transformed, log_det = eqx.filter_jit(bijection.transform_and_log_det)(x)
    assert jnp.allclose(transformed, expected)
    assert jnp.allclose(log_det, expected_log_det)
    restored, inverse_log_det = eqx.filter_jit(bijection.inverse_and_log_det)(expected)
    assert jnp.allclose(restored, x)
    assert jnp.allclose(inverse_log_det, -expected_log_det)
