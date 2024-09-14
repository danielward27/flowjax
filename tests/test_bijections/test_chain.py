"Tests for bijection.chain module."
from functools import partial

import equinox as eqx
import jax.numpy as jnp
import pytest
from jax import random

from flowjax.bijections import Affine, Chain, Coupling, Flip, Permute, Scan


def test_chain_dunders():
    bijection = Chain([Flip((2,)), Permute(jnp.array([0, 1]))])
    assert len(bijection) == 2
    assert isinstance(bijection[0], Flip)
    assert isinstance(bijection[1], Permute)
    assert isinstance(bijection[:], Chain)


def test_merge_chains():
    scale = jnp.array([0.5, 2])
    bijections = [
        Chain([Affine(scale=scale), Chain([Flip((2,)), Affine(scale=scale)])]),
        Flip((2,)),
    ]

    chain = Chain(bijections)

    x = jnp.arange(2)
    assert chain.transform(x) == pytest.approx(chain.merge_chains().transform(x))
    assert not any(isinstance(b, Chain) for b in chain.merge_chains().bijections)


DIM = 4
COND_DIM = 5
NUM_LAYERS = 3
keys = random.split(random.key(0), NUM_LAYERS)

make_coupling_layer = partial(
    Coupling,
    transformer=Affine(),
    untransformed_dim=DIM // 2,
    dim=DIM,
    cond_dim=COND_DIM,
    nn_width=10,
    nn_depth=3,
)

affine_params = jnp.ones((NUM_LAYERS, DIM))

test_cases = {
    "Affine": lambda: (
        Chain([Affine(p) for p in affine_params]),
        Scan(eqx.filter_vmap(Affine)(affine_params)),
    ),
    "Coupling": lambda: (
        Chain([make_coupling_layer(k) for k in keys]),
        Scan(eqx.filter_vmap(make_coupling_layer)(keys)),
    ),
}


@pytest.mark.parametrize("name", test_cases.keys())
def test_scan(name):
    "Check Chain and Scan give consistent results."
    chain, scan = test_cases[name]()
    x = jnp.ones(DIM)
    condition = jnp.ones(chain.cond_shape) if chain.cond_shape is not None else None
    expected = pytest.approx(chain.transform(x, condition))
    assert expected == scan.transform(x, condition)

    expected = pytest.approx(chain.inverse(x, condition))
    realised = scan.inverse(x, condition)
    assert expected == realised

    expected = chain.transform_and_log_det(x, condition)
    realised = scan.transform_and_log_det(x, condition)
    assert jnp.all(
        jnp.array(
            [pytest.approx(a) == b for (a, b) in zip(expected, realised, strict=True)],
        ),
    )
