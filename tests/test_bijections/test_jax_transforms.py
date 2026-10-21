import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import paramax
import pytest
from jax.tree_util import tree_map

from flowjax.bijections import AdditiveCondition, Affine, MaskedAutoregressive, Vmap


def test_vmap_uneven_init():
    "Tests adding a batch dimension to a particular leaf (parameter array)."
    bijection = Affine(jnp.zeros(()), jnp.ones(()))
    bijection = eqx.tree_at(lambda bij: bij.loc, bijection, jnp.arange(3))
    in_axes = tree_map(lambda _: None, paramax.unwrap(bijection))
    in_axes = eqx.tree_at(lambda bij: bij.loc, in_axes, 0, is_leaf=lambda x: x is None)
    bijection = Vmap(bijection, in_axes=in_axes)

    assert bijection.shape == (3,)
    assert bijection.bijection.loc.shape == (3,)
    assert paramax.unwrap(bijection.bijection.scale).shape == ()

    x = jnp.ones(3)
    expected = x + jnp.arange(3)
    assert bijection.transform(x) == pytest.approx(expected)


def test_vmap_error_with_unwrappable():
    bijection = Affine(jnp.zeros(1), jnp.ones(1))
    in_axes = tree_map(eqx.is_array, bijection)
    with pytest.raises(ValueError, match="unwrappable"):
        bijection = Vmap(bijection, in_axes=in_axes)


def test_vmap_condition_only():
    bijection = MaskedAutoregressive(
        jr.key(0),
        transformer=Affine(),
        dim=3,
        cond_dim=4,
        nn_width=10,
        nn_depth=1,
    )

    with pytest.raises(
        ValueError,
        match="Either axis_size or in_axes must be provided.",
    ):
        bijection = Vmap(bijection, in_axes_condition=0)

    bijection = Vmap(bijection, in_axes_condition=1, axis_size=10)
    assert bijection.shape == (10, 3)
    assert bijection.cond_shape == (4, 10)

    x = jnp.ones((10, 3))
    condition = jnp.linspace(-2, 2, 10 * 4).reshape(4, 10)
    y = bijection.transform(x, condition)
    x_reconstructed = bijection.inverse(y, condition)
    assert x_reconstructed == pytest.approx(x)


@pytest.mark.parametrize(
    ("shape", "axis"),
    [((3,), -1), ((3, 4), -1), ((3, 4), -2), ((3, 4), -3), ((), -1), ((3,), 1)],
)
def test_vmap_condition_axis_matches_array_stack(shape, axis):
    bijection = Vmap(
        AdditiveCondition(lambda condition: condition, shape, shape),
        axis_size=2,
        in_axes_condition=axis,
    )
    conditions = [jnp.ones(shape), jnp.full(shape, 2.0)]
    condition = jnp.stack(conditions, axis=axis)
    x = jnp.zeros((2, *shape))
    expected = jnp.stack(conditions)

    assert bijection.cond_shape == condition.shape
    transformed, log_det = eqx.filter_jit(bijection.transform_and_log_det)(x, condition)
    assert jnp.allclose(transformed, expected)
    assert log_det == 0
    restored, inverse_log_det = eqx.filter_jit(bijection.inverse_and_log_det)(
        expected, condition
    )
    assert jnp.allclose(restored, x)
    assert inverse_log_det == 0
