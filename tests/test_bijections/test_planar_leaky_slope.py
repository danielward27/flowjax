import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from flowjax.bijections import Planar
from flowjax.distributions import StandardNormal, Transformed


@pytest.mark.parametrize("slope", [3.0, 5.0])
@pytest.mark.parametrize("jit", [False, True])
def test_planar_large_leaky_slope_roundtrip(slope, jit):
    bijection = Planar(jr.key(0), dim=1, negative_slope=slope)
    transform = bijection.transform_and_log_det
    inverse = bijection.inverse_and_log_det
    if jit:
        transform = eqx.filter_jit(transform)
        inverse = eqx.filter_jit(inverse)
    for x in (jnp.array([-1.0]), jnp.array([1.0])):
        y, log_det = transform(x)
        recovered, inverse_log_det = inverse(y)
        assert jnp.allclose(recovered, x, atol=1e-5)
        jacobian = jax.jacfwd(bijection.transform)(x)
        assert jnp.linalg.det(jacobian) > 0
        assert jnp.allclose(log_det, jnp.linalg.slogdet(jacobian)[1], atol=1e-5)
        assert jnp.allclose(log_det + inverse_log_det, 0, atol=1e-5)


@pytest.mark.parametrize("conditional", [False, True])
def test_planar_large_leaky_slope_sample_log_prob(conditional):
    bijection = Planar(
        jr.key(0),
        dim=1,
        negative_slope=3.0,
        cond_dim=2 if conditional else None,
        width_size=4,
        depth=1,
    )
    dist = Transformed(StandardNormal((1,)), bijection)
    condition = jnp.array([0.2, -0.3]) if conditional else None
    samples, log_probs = eqx.filter_jit(dist.sample_and_log_prob)(
        jr.key(1),
        (32,),
        condition=condition,
    )
    actual = eqx.filter_jit(dist.log_prob)(samples, condition=condition)
    assert jnp.all(jnp.isfinite(log_probs))
    assert jnp.allclose(actual, log_probs, atol=1e-4)
