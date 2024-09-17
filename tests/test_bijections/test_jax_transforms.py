import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.tree_util import tree_map

from flowjax.bijections import Affine, MaskedAutoregressive, Vmap
from flowjax.wrappers import unwrap


def test_vmap_uneven_init():
    "Tests adding a batch dimension to a particular leaf (parameter array)."
    bijection = Affine(jnp.zeros(()), jnp.ones(()))
    bijection = eqx.tree_at(lambda bij: bij.loc, bijection, jnp.arange(3))
    in_axes = tree_map(lambda _: None, unwrap(bijection))
    in_axes = eqx.tree_at(lambda bij: bij.loc, in_axes, 0, is_leaf=lambda x: x is None)
    bijection = Vmap(bijection, in_axes=in_axes)

    assert bijection.shape == (3,)
    assert bijection.bijection.loc.shape == (3,)
    assert unwrap(bijection.bijection.scale).shape == ()

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
