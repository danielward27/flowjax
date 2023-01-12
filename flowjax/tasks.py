"Example tasks"

import jax.random as jr
import jax.numpy as jnp


def two_moons(key, n_samples, noise_std=0.2):
    "Two moon distribution."
    angle_key, noise_key = jr.split(key)
    angle = jr.uniform(angle_key, (n_samples,)) * 2 * jnp.pi
    x = 2 * jnp.cos(angle)
    y = 2 * jnp.sin(angle)
    x = jnp.where(angle > jnp.pi, x + 1, x - 1)
    noise = jr.normal(noise_key, (n_samples, 2)) * noise_std
    return jnp.stack([x, y], axis=1) + noise
