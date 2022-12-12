"General tests for bijections (including transformers)."
import pytest
from jax import random
import jax.numpy as jnp
import jax
from flowjax.bijections.transformers import (
    AffineTransformer,
    RationalQuadraticSplineTransformer,
)
import equinox as eqx

from flowjax.bijections import (
    Affine,
    TriangularAffine,
    Coupling,
    MaskedAutoregressive,
    Tanh,
    Flip,
    Permute,
    BlockAutoregressiveNetwork,
    TransformerToBijection,
    AdditiveLinearCondition,
    Partial,
    EmbedCondition,
)


dim = 5
cond_dim = 2
key = random.PRNGKey(0)
pos_def_triangles = jnp.full((dim, dim), 0.5) + jnp.diag(jnp.ones(dim))

bijections = {
    "Flip": Flip(),
    "Permute": Permute(jnp.flip(jnp.arange(dim))),
    "Partial (int)": Partial(Affine(2, 2), 0),
    "Partial (bool array)": Partial(Flip(), jnp.array([True, False] * 2 + [True])),
    "Partial (int array)": Partial(Flip(), jnp.array([0, 0, 4, 3])),
    "Partial (slice)": Partial(Flip(), slice(1, 3)),
    "Affine": Affine(jnp.ones(dim), jnp.full(dim, 2)),
    "Tanh": Tanh(),
    "TriangularAffine (lower)": TriangularAffine(jnp.arange(dim), pos_def_triangles),
    "TriangularAffine (upper)": TriangularAffine(
        jnp.arange(dim), pos_def_triangles, lower=False
    ),
    "TriangularAffine (weight_norm)": TriangularAffine(
        jnp.arange(dim), pos_def_triangles, weight_normalisation=True
    ),
    "Coupling (unconditional)": Coupling(
        key,
        AffineTransformer(),
        d=dim // 2,
        D=dim,
        cond_dim=0,
        nn_width=10,
        nn_depth=2,
    ),
    "Coupling (conditional)": Coupling(
        key,
        AffineTransformer(),
        d=dim // 2,
        D=dim,
        cond_dim=cond_dim,
        nn_width=10,
        nn_depth=2,
    ),
    "MaskedAutoregressive_Affine (unconditional)": MaskedAutoregressive(
        key, AffineTransformer(), cond_dim=0, dim=dim, nn_width=10, nn_depth=2
    ),
    "MaskedAutoregressive_Affine (conditional)": MaskedAutoregressive(
        key, AffineTransformer(), cond_dim=cond_dim, dim=dim, nn_width=10, nn_depth=2
    ),
    "MaskedAutoregressive_RationalQuadraticSpline (unconditional)": MaskedAutoregressive(
        key,
        RationalQuadraticSplineTransformer(5, 3),
        dim=dim,
        cond_dim=0,
        nn_width=10,
        nn_depth=2,
    ),
    "MaskedAutoregressive_RationalQuadraticSpline (conditional)": MaskedAutoregressive(
        key,
        RationalQuadraticSplineTransformer(5, 3),
        dim=dim,
        cond_dim=cond_dim,
        nn_width=10,
        nn_depth=2,
    ),
    "BlockAutoregressiveNetwork": BlockAutoregressiveNetwork(
        key, dim=dim, cond_dim=0, block_dim=3, depth=1
    ),
    "BlockAutoregressiveNetwork (conditional)": BlockAutoregressiveNetwork(
        key, dim=dim, cond_dim=cond_dim, block_dim=3, depth=1
    ),
    "AdditiveLinearCondition": AdditiveLinearCondition(
        random.uniform(key, (dim, cond_dim))
    ),
    "EmbedCondition": EmbedCondition(
        BlockAutoregressiveNetwork(key, dim=dim, cond_dim=1, block_dim=3, depth=1),
        eqx.nn.MLP(2, 1, 3, 1, key=key),
        cond_dim,
    ),
}

transformers = {
    "AffineTransformer": AffineTransformer(),
    "RationalQuadraticSplineTransformer": RationalQuadraticSplineTransformer(K=5, B=3),
}

transformers = {
    k: TransformerToBijection(b, params=random.normal(key, (b.num_params(dim),)))
    for k, b in transformers.items()
}

bijections = bijections | transformers


@pytest.mark.parametrize("bijection", bijections.values(), ids=bijections.keys())
def test_transform_inverse(bijection):
    """Tests transform and inverse methods."""
    x = random.normal(random.PRNGKey(0), (dim,))
    if bijection.cond_dim > 0:
        cond = random.normal(random.PRNGKey(0), (bijection.cond_dim,))
    else:
        cond = None
    y = bijection.transform(x, cond)
    try:
        x_reconstructed = bijection.inverse(y, cond)
        assert x == pytest.approx(x_reconstructed, abs=1e-4)
    except NotImplementedError:
        pass


@pytest.mark.parametrize("bijection", bijections.values(), ids=bijections.keys())
def test_transform_inverse_and_log_dets(bijection):
    """Tests the transform_and_log_abs_det_jacobian and inverse_and_log_abs_det_jacobian methods,
    by 1) checking invertibility and 2) comparing log dets to those obtained with
    automatic differentiation."""
    x = random.normal(random.PRNGKey(0), (dim,))

    if bijection.cond_dim > 0:
        cond = random.normal(random.PRNGKey(0), (bijection.cond_dim,))
    else:
        cond = None
    auto_jacobian = jax.jacobian(bijection.transform)(x, cond)
    auto_log_det = jnp.log(jnp.abs(jnp.linalg.det(auto_jacobian)))
    y, logdet = bijection.transform_and_log_abs_det_jacobian(x, cond)
    assert logdet == pytest.approx(auto_log_det, abs=1e-4)

    try:
        x_reconstructed, logdetinv = bijection.inverse_and_log_abs_det_jacobian(y, cond)
        assert logdetinv == pytest.approx(-auto_log_det, abs=1e-4)
        assert x == pytest.approx(x_reconstructed, abs=1e-4)

    except NotImplementedError:
        pass
