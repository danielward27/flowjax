import pytest
from flowjax.utils import broadcast_except_last
import jax.numpy as jnp

def test_broadcast():
    # Matrices
    size_pairs = [((5,2), (5,3)), ((1,2), (5,3)), ((5,2), (1,3)), ((2,), (5,3)), ((5,2), (3,))]
    out_sizes = [((5,2), (5,3))] * len(size_pairs)

    for in_s, out_s in zip(size_pairs, out_sizes):
        a,b = broadcast_except_last(jnp.ones(in_s[0]), jnp.ones(in_s[1]))
        assert (a.shape, b.shape) == out_s
        