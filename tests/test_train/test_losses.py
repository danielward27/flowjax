import jax.numpy as jnp
import jax.random as jr

from flowjax.train.losses import _get_contrastive_idxs


def test_get_contrastive_idxs():
    key = jr.key(0)
    batch_size = 5

    for _ in range(5):
        key, subkey = jr.split(key)
        idxs = _get_contrastive_idxs(subkey, batch_size=batch_size, n_contrastive=4)
        for i, row in enumerate(idxs):
            assert i not in row

        assert jnp.all(idxs < batch_size)
