"General tests for bijections (including transformers)."
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from flowjax.bijections import (
    AdditiveLinearCondition,
    Affine,
    BlockAutoregressiveNetwork,
    Chain,
    Concatenate,
    Coupling,
    EmbedCondition,
    Exp,
    Flip,
    MaskedAutoregressive,
    Partial,
    Permute,
    RationalQuadraticSpline,
    Scan,
    Stack,
    Tanh,
    TanhLinearTails,
    TriangularAffine,
)


DIM = 5
COND_DIM = 2
KEY = jr.PRNGKey(0)
POS_DEF_TRAINGLES = jnp.full((DIM, DIM), 0.5) + jnp.diag(jnp.ones(DIM))


def get_maf_layer(key):
    """Get a masked autoregressive flow layer."""
    return MaskedAutoregressive(
        key, Affine(), DIM, cond_dim=COND_DIM, nn_width=5, nn_depth=5
    )


bijections = {
    "Flip": Flip(),
    "Permute": Permute(jnp.flip(jnp.arange(DIM))),
    "Permute (3D)": Permute(
        jnp.reshape(jr.permutation(KEY, jnp.arange(2 * 3 * 4)), (2, 3, 4))
    ),
    "Partial (int)": Partial(Affine(jnp.array(2), jnp.array(2)), 0),
    "Partial (bool array)": Partial(Flip(), jnp.array([True, False] * 2 + [True])),
    "Partial (int array)": Partial(Flip(), jnp.array([0, 4])),
    "Partial (slice)": Partial(Affine(jnp.zeros(3)), slice(0, 3)),
    "Affine": Affine(jnp.ones(DIM), jnp.full(DIM, 2)),
    "Tanh": Tanh(),
    "Exp": Exp(),
    "TanhLinearTails": TanhLinearTails(1),
    "TriangularAffine (lower)": TriangularAffine(jnp.arange(DIM), POS_DEF_TRAINGLES),
    "TriangularAffine (upper)": TriangularAffine(
        jnp.arange(DIM), POS_DEF_TRAINGLES, lower=False
    ),
    "TriangularAffine (weight_norm)": TriangularAffine(
        jnp.arange(DIM), POS_DEF_TRAINGLES, weight_normalisation=True
    ),
    "RationalQuadraticSpline": RationalQuadraticSpline(knots=4, interval=1, shape=(5,)),
    "Coupling (unconditional)": Coupling(
        KEY,
        Affine(),
        d=DIM // 2,
        D=DIM,
        cond_dim=None,
        nn_width=10,
        nn_depth=2,
    ),
    "Coupling (conditional)": Coupling(
        KEY,
        Affine(),
        d=DIM // 2,
        D=DIM,
        cond_dim=COND_DIM,
        nn_width=10,
        nn_depth=2,
    ),
    "MaskedAutoregressive_Affine (unconditional)": MaskedAutoregressive(
        KEY, Affine(), cond_dim=0, dim=DIM, nn_width=10, nn_depth=2
    ),
    "MaskedAutoregressive_Affine (conditional)": MaskedAutoregressive(
        KEY, Affine(), cond_dim=COND_DIM, dim=DIM, nn_width=10, nn_depth=2
    ),
    "MaskedAutoregressive_RationalQuadraticSpline (unconditional)": MaskedAutoregressive(
        KEY,
        RationalQuadraticSpline(5, 3),
        dim=DIM,
        cond_dim=0,
        nn_width=10,
        nn_depth=2,
    ),
    "BlockAutoregressiveNetwork (unconditional)": BlockAutoregressiveNetwork(
        KEY, dim=DIM, cond_dim=0, block_dim=3, depth=1
    ),
    "BlockAutoregressiveNetwork (conditional)": BlockAutoregressiveNetwork(
        KEY, dim=DIM, cond_dim=COND_DIM, block_dim=3, depth=1
    ),
    "AdditiveLinearCondition": AdditiveLinearCondition(
        jr.uniform(KEY, (DIM, COND_DIM))
    ),
    "EmbedCondition": EmbedCondition(
        BlockAutoregressiveNetwork(KEY, dim=DIM, cond_dim=1, block_dim=3, depth=1),
        eqx.nn.MLP(2, 1, 3, 1, key=KEY),
        (COND_DIM,),  # Raw
    ),
    "Chain": Chain([Flip(), Affine(jnp.ones(DIM), jnp.full(DIM, 2))]),
    "Scan": Scan(eqx.filter_vmap(get_maf_layer)(jr.split(KEY, 3))),
    "Concatenate": Concatenate([Affine(jnp.ones(3)), Tanh(shape=(3,))]),
    "ConcatenateAxis1": Concatenate(
        [Affine(jnp.ones((3, 3))), Tanh(shape=((3, 3)))], axis=1
    ),
    "ConcatenateAxis-1": Concatenate(
        [Affine(jnp.ones((3, 3))), Tanh(shape=((3, 3)))], axis=-1
    ),
    "Stack": Stack([Tanh(()), Affine(), Tanh(())]),
    "StackAxis1": Stack([Tanh((2,)), Affine(jnp.ones(2)), Tanh((2,))], axis=1),
    "StackAxis-1": Stack(
        [Affine(jr.uniform(k, (1, 2, 3))) for k in jr.split(KEY, 3)], axis=-1
    ),
}


@pytest.mark.parametrize("bijection", bijections.values(), ids=bijections.keys())
def test_transform_inverse(bijection):
    """Tests transform and inverse methods."""
    shape = bijection.shape if bijection.shape is not None else (DIM,)
    x = jr.normal(jr.PRNGKey(0), shape)
    if bijection.cond_shape is not None:
        cond = jr.normal(jr.PRNGKey(0), bijection.cond_shape)
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
    shape = bijection.shape if bijection.shape is not None else (DIM,)
    x = jr.normal(jr.PRNGKey(0), shape)

    if bijection.cond_shape is not None:
        cond = jr.normal(jr.PRNGKey(0), bijection.cond_shape)
    else:
        cond = None

    # We flatten the function so auto_jacobian is calculated correctly
    def flat_transform(x_flat, cond):
        x = x_flat.reshape(shape)
        y = bijection.transform(x, cond)
        return y.ravel()

    auto_jacobian = jax.jacobian(flat_transform)(x.ravel(), cond)
    auto_log_det = jnp.log(jnp.abs(jnp.linalg.det(auto_jacobian)))
    y, logdet = bijection.transform_and_log_abs_det_jacobian(x, cond)
    assert logdet == pytest.approx(auto_log_det, abs=1e-4)

    try:
        x_reconstructed, logdetinv = bijection.inverse_and_log_abs_det_jacobian(y, cond)
        assert logdetinv == pytest.approx(-auto_log_det, abs=1e-4)
        assert x == pytest.approx(x_reconstructed, abs=1e-4)

    except NotImplementedError:
        pass
