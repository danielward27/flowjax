import pytest
from jax import random
import jax.numpy as jnp
from flowjax.bijections.coupling import Coupling
from flowjax.bijections.masked_autoregressive import MaskedAutoregressive
from flowjax.bijections.transformers import AffineTransformer
from flowjax.bijections.utils import Flip, Permute
from flowjax.bijections.transformers import AffineTransformer, RationalQuadraticSplineTransformer
from flowjax.bijections.affine import Affine, TriangularAffine

transformers = {
    "AffineTransformer": AffineTransformer(),
    "RationalQuadraticSplineTransformer": RationalQuadraticSplineTransformer(K=5, B=3),
}

@pytest.mark.parametrize("bijection", transformers.values(), ids=transformers.keys())
def test_transformer_invertibility(bijection):
    d = 5
    x = random.normal(random.PRNGKey(0), (d,))
    params = random.normal(random.PRNGKey(1), (bijection.num_params(d),))
    args = bijection.get_args(params)
    y = bijection.transform(x, *args)
    x_reconstructed = bijection.inverse(y, *args)

    assert jnp.any(x != y)  # Check change occurs
    assert x == pytest.approx(x_reconstructed, abs=1e-5)  # Check invertibility

    # Check log dets
    y, log_det1 = bijection.transform_and_log_abs_det_jacobian(x, *args)
    x_reconstructed, log_det2 = bijection.inverse_and_log_abs_det_jacobian(y, *args)
    
    assert x == pytest.approx(x_reconstructed, abs=1e-5)  # Check invertibility
    assert log_det1 == pytest.approx(-log_det2, abs=1e-5)


dim = 5
cond_dim = 2
key = random.PRNGKey(0)
pos_def_triangles = jnp.full((dim,dim), 0.5) + jnp.diag(jnp.ones(dim))

bijections = {
    "Flip": Flip(),
    "Permute": Permute(jnp.flip(jnp.arange(dim))),
    "Affine": Affine(jnp.ones(dim), jnp.full(dim, 2)),
    "TriangularAffine (lower)": TriangularAffine(jnp.arange(dim), pos_def_triangles),
    "TriangularAffine (upper)": TriangularAffine(jnp.arange(dim), pos_def_triangles, lower=False),
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
        key, RationalQuadraticSplineTransformer(5, 3), cond_dim=0, dim=dim, nn_width=10, nn_depth=2
    ),
    "MaskedAutoregressive_RationalQuadraticSpline (conditional)": MaskedAutoregressive(
        key, RationalQuadraticSplineTransformer(5, 3), cond_dim=cond_dim, dim=dim, nn_width=10, nn_depth=2
    ),
}


@pytest.mark.parametrize("bijection", bijections.values(), ids=bijections.keys())
def test_bijection_invertibility(bijection):
    x = random.normal(random.PRNGKey(0), (dim,))
    if bijection.cond_dim > 0:
        cond = random.normal(random.PRNGKey(0), (bijection.cond_dim,))
    else:
        cond = None
    y = bijection.transform(x, cond)
    x_reconstructed = bijection.inverse(y, cond)
    assert x == pytest.approx(x_reconstructed, abs=1e-5)

    y, log_det1 = bijection.transform_and_log_abs_det_jacobian(x, cond)
    x_reconstructed, log_det2 = bijection.inverse_and_log_abs_det_jacobian(y, cond)
    
    assert x == pytest.approx(x_reconstructed, abs=1e-5)
    assert log_det1 == pytest.approx(-log_det2, abs=1e-5)
