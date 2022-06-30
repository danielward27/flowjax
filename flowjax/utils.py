import jax.numpy as jnp

def broadcast_except_last(x: jnp.ndarray, condition: jnp.ndarray):
    "Broadcast arrays, excluding last axis, returning at least 2d results."
    x, condition = jnp.atleast_2d(x), jnp.atleast_2d(condition)
    s = jnp.broadcast_shapes(x.shape[:-1], condition.shape[:-1])
    x = jnp.broadcast_to(x, s + (x.shape[-1],))
    condition = jnp.broadcast_to(condition, s + (condition.shape[-1],))
    return x, condition
    