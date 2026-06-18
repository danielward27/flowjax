import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import vmap
from jax.tree_util import tree_map, tree_leaves

from flowjax.bijections import RationalQuadraticSpline


@pytest.mark.parametrize("interval", [3, (-4, 5)])
def test_RationalQuadraticSpline_tails(interval):
    key = jr.key(0)
    spline = RationalQuadraticSpline(knots=10, interval=interval)

    # Change to random initialisation, rather than identity.
    spline = tree_map(
        lambda x: jr.normal(key, x.shape) if eqx.is_inexact_array(x) else x,
        spline,
    )

    x = jr.uniform(key, (5,), minval=spline.interval[0], maxval=spline.interval[1])
    y = vmap(spline.transform)(x)
    assert pytest.approx(x, abs=1e-5) != y

    # Outside interval, default to identity
    x = jnp.array([spline.interval[0] - 1, spline.interval[1] + 1])
    y = vmap(spline.transform)(x)
    assert pytest.approx(x, abs=1e-5) == y


@pytest.mark.parametrize("interval", [3, (-4, 5)])
def test_RationalQuadraticSpline_init(interval):
    # Test it is initialized at the identity
    x = jnp.array([-7, 0.1, 2, 1])
    spline = RationalQuadraticSpline(knots=10, interval=interval)
    y = vmap(spline.transform)(x)
    assert pytest.approx(x, abs=1e-6) == y

@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_RationalQuadraticSpline_out_of_bounds_grad_nan(direction):
    """Checks that gradients do not contain NaNs when evaluating the forward 
    and inverse passes on out-of-bounds inputs."""
    
    # 1. Initialize RQS
    spline = RationalQuadraticSpline(knots=4, interval=1.0)

    # 2. Partition and add truly random noise to simulate a "trained" state
    params, static = eqx.partition(spline, eqx.is_inexact_array)
    key = jr.key(42)
    
    def add_noise(x):
        nonlocal key
        if x is not None:
            key, subkey = jr.split(key)
            return x + jr.normal(subkey, x.shape) * 1.5
        return x
        
    params_noisy = tree_map(add_noise, params)
    spline_trained = eqx.combine(params_noisy, static)

    # 3. Provide an out-of-bounds input (interval is 1.0, so 50.0 is way out)
    val = jnp.array(50.0)

    # 4. Define a dummy loss evaluating the requested direction
    def loss(model, v):
        if direction == "forward":
            out, _ = model.transform_and_log_det(v)
        else:
            out, _ = model.inverse_and_log_det(v)
        return out.sum()

    # 5. Compute gradients
    _, grad = eqx.filter_value_and_grad(loss)(spline_trained, val)

    # 6. Check for NaNs
    nan_found = any(
        jnp.any(jnp.isnan(leaf)) 
        for leaf in tree_leaves(grad) if eqx.is_array(leaf)
    )
    
    assert not nan_found, f"NaNs detected in gradients for out-of-bounds {direction} pass"