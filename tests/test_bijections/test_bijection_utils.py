from flowjax.bijections.utils import Chain, Flip, Permute, intertwine_flip, intertwine_random_permutation
import pytest
from jax import random
import jax.numpy as jnp


perm = jnp.array([0,1])
test_cases = {
    # {name: (inputs, expected)}
    "Len 3": ([Permute(perm) for _ in range(3)], [Permute, Flip, Permute, Flip, Permute]),
    "Len 1": ([Permute(perm)], [Permute])
}

@pytest.mark.parametrize("bijections,expected", test_cases.values(), ids=test_cases.keys())
def test_intertwine_flip(bijections, expected):
    "Intertwine flips (between permutation bijections)"
    bijections = intertwine_flip(bijections)
    assert [type(b) == ex for b, ex in zip(bijections, expected)]


test_cases = {
    "Len 3": ([Flip() for _ in range(3)], [Flip, Permute, Flip, Permute, Flip]),
    "Len 1": ([Flip()], [Flip])
}

@pytest.mark.parametrize("bijections,expected", test_cases.values(), ids=test_cases.keys())
def test_intertwine_random_permutation(bijections, expected):
    "Intertwine permutations (between Flip bijections)"
    bijections = intertwine_random_permutation(random.PRNGKey(0), bijections, dim=2)
    assert [type(b) == ex for b, ex in zip(bijections, expected)]


def test_chain_dunders():
    b = Chain([Flip(), Permute(jnp.array([0,1]))])
    assert len(b) == 2
    assert isinstance(b[0], Flip)
    assert isinstance(b[1], Permute)
    assert isinstance(b[:], Chain)
