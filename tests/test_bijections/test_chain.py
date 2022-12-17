# %%
from functools import partial

import equinox as eqx
import jax.numpy as jnp
import pytest
from jax import random

from flowjax.bijections import Affine, Chain, Coupling, Flip, Permute, ScannableChain
from flowjax.bijections.transformers import AffineTransformer


def test_chain_dunders():
    b = Chain([Flip(), Permute(jnp.array([0, 1]))])
    assert len(b) == 2
    assert isinstance(b[0], Flip)
    assert isinstance(b[1], Permute)
    assert isinstance(b[:], Chain)


dim = 4
cond_dim = 5
num_layers = 3
keys = random.split(random.PRNGKey(0), num_layers)


make_coupling_layer = partial(
    Coupling,
    transformer=AffineTransformer(),
    d=dim // 2,
    D=dim,
    cond_dim=cond_dim,
    nn_width=10,
    nn_depth=3,
)


params = jnp.ones((num_layers, dim))
affine_chain = Chain([Affine(p) for p in params])
affine_scan = ScannableChain(eqx.filter_vmap(Affine)(params))

coupling_chain = Chain([make_coupling_layer(k) for k in keys])
coupling_scan = ScannableChain(eqx.filter_vmap(make_coupling_layer)(keys))

test_cases = {
    "Affine": (affine_scan, affine_chain),
    "Coupling": (coupling_scan, coupling_chain),
}


@pytest.mark.parametrize("scan,chain", test_cases.values(), ids=test_cases.keys())
def test_scannable_chain(scan, chain):
    "Check Chain and ScannableChain give consistent results."
    x = jnp.ones(dim)
    condition = jnp.ones(chain.cond_dim) if chain.cond_dim > 0 else None
    expected = pytest.approx(chain.transform(x, condition))
    assert expected == scan.transform(x, condition)

    expected = pytest.approx(chain.inverse(x, condition))
    realised = scan.inverse(x, condition)
    assert expected == realised

    expected = chain.transform_and_log_abs_det_jacobian(x, condition)
    realised = scan.transform_and_log_abs_det_jacobian(x, condition)
    assert jnp.all(
        jnp.array([pytest.approx(a) == b for (a, b) in zip(expected, realised)])
    )
