"General tests for bijections (including transformers)."
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from flowjax.bijections import (
    AbstractBijection,
    AdditiveCondition,
    Affine,
    BlockAutoregressiveNetwork,
    Chain,
    Concatenate,
    Coupling,
    EmbedCondition,
    Exp,
    Flip,
    Identity,
    LeakyTanh,
    Loc,
    MaskedAutoregressive,
    Partial,
    Permute,
    Planar,
    RationalQuadraticSpline,
    Reshape,
    Scale,
    Scan,
    SoftPlus,
    Stack,
    Tanh,
    TriangularAffine,
    Vmap,
)

DIM = 3
COND_DIM = 2
KEY = jr.PRNGKey(0)


bijections = {
    "Identity": Identity((DIM,)),
    "Flip": Flip((DIM,)),
    "Permute": Permute(jnp.flip(jnp.arange(DIM))),
    "Permute (3D)": Permute(
        jnp.reshape(jr.permutation(KEY, jnp.arange(2 * 3 * 4)), (2, 3, 4)),
    ),
    "Partial (int)": Partial(Affine(jnp.array(2), jnp.array(2)), 0, (DIM,)),
    "Partial (bool array)": Partial(
        Flip((2,)),
        jnp.array([True, False, True]),
        (DIM,),
    ),
    "Partial (int array)": Partial(Flip((2,)), jnp.array([0, 2]), (DIM,)),
    "Partial (slice)": Partial(Affine(jnp.zeros(2)), slice(0, 2), (DIM,)),
    "Affine": Affine(jnp.ones(DIM), jnp.full(DIM, 2)),
    "Affine (pos and neg scales)": eqx.tree_at(
        lambda aff: aff.scale, Affine(scale=jnp.ones(3)), jnp.array([-1, 1, -2])
    ),
    "Tanh": Tanh((DIM,)),
    "LeakyTanh": LeakyTanh(1, (DIM,)),
    "LeakyTanh (broadcast max_val)": LeakyTanh(1, (2, 3)),
    "Loc": Loc(jnp.arange(DIM)),
    "Exp": Exp((DIM,)),
    "SoftPlus": SoftPlus((DIM,)),
    "TriangularAffine (lower)": TriangularAffine(
        jnp.arange(DIM),
        jnp.full((DIM, DIM), 0.5),
    ),
    "TriangularAffine (upper)": TriangularAffine(
        jnp.arange(DIM),
        jnp.full((DIM, DIM), 0.5),
        lower=False,
    ),
    "TriangularAffine (pos and neg diag)": eqx.tree_at(
        lambda triaff: triaff.triangular,
        TriangularAffine(
            jnp.arange(3),
            jnp.diag(jnp.ones(3)),
        ),
        jnp.diag(jnp.array([-1, 2, -3])),
    ),
    "RationalQuadraticSpline": RationalQuadraticSpline(knots=4, interval=1),
    "Coupling (unconditional)": Coupling(
        KEY,
        transformer=Affine(),
        untransformed_dim=DIM // 2,
        dim=DIM,
        nn_width=5,
        nn_depth=2,
    ),
    "Coupling (conditional)": Coupling(
        KEY,
        transformer=Affine(),
        untransformed_dim=DIM // 2,
        dim=DIM,
        cond_dim=COND_DIM,
        nn_width=5,
        nn_depth=2,
    ),
    "MaskedAutoregressive_Affine (unconditional)": MaskedAutoregressive(
        KEY,
        transformer=Affine(),
        dim=DIM,
        nn_width=5,
        nn_depth=2,
    ),
    "MaskedAutoregressive_Affine (conditional)": MaskedAutoregressive(
        KEY,
        transformer=Affine(),
        cond_dim=COND_DIM,
        dim=DIM,
        nn_width=5,
        nn_depth=2,
    ),
    "MaskedAutoregressiveRationalQuadraticSpline (unconditional)": MaskedAutoregressive(
        KEY,
        transformer=RationalQuadraticSpline(knots=5, interval=3),
        dim=DIM,
        nn_width=10,
        nn_depth=2,
    ),
    "BlockAutoregressiveNetwork (unconditional)": BlockAutoregressiveNetwork(
        KEY,
        dim=DIM,
        block_dim=3,
        depth=1,
    ),
    "BlockAutoregressiveNetwork (conditional)": BlockAutoregressiveNetwork(
        KEY,
        dim=DIM,
        cond_dim=COND_DIM,
        block_dim=3,
        depth=1,
    ),
    "AdditiveCondtition": AdditiveCondition(
        lambda condition: jnp.arange(DIM) * jnp.sum(condition),
        (DIM,),
        (COND_DIM,),
    ),
    "EmbedCondition": EmbedCondition(
        BlockAutoregressiveNetwork(KEY, dim=DIM, cond_dim=1, block_dim=3, depth=1),
        eqx.nn.MLP(2, 1, 3, 1, key=KEY),
        (COND_DIM,),  # Raw
    ),
    "Chain": Chain([Flip((DIM,)), Affine(jnp.ones(DIM), jnp.full(DIM, 2))]),
    "Scan": Scan(eqx.filter_vmap(Affine)(jnp.ones((2, DIM)))),
    "Scale": Scale(jnp.full(DIM, 2)),
    "Scale (pos and neg scales)": eqx.tree_at(
        lambda scale: scale.scale,
        Scale(jnp.ones(3)),
        jnp.array([-1, 2, -3]),
    ),
    "Concatenate": Concatenate([Affine(jnp.ones(DIM)), Tanh(shape=(DIM,))]),
    "ConcatenateAxis1": Concatenate(
        [Affine(jnp.ones((3, 3))), Tanh(shape=((3, 3)))],
        axis=1,
    ),
    "ConcatenateAxis-1": Concatenate(
        [Affine(jnp.ones((3, 3))), Tanh(shape=((3, 3)))],
        axis=-1,
    ),
    "Stack": Stack([Tanh(()), Affine(), Tanh(())]),
    "StackAxis1": Stack([Tanh((2,)), Affine(jnp.ones(2)), Tanh((2,))], axis=1),
    "StackAxis-1": Stack(
        [Affine(jr.uniform(k, (1, 2, 3))) for k in jr.split(KEY, 3)],
        axis=-1,
    ),
    "Planar": Planar(
        KEY,
        dim=DIM,
    ),
    "Vmap (broadcast params)": Vmap(Affine(1, 2), axis_size=10),
    "Vmap (vectorize params)": Vmap(
        eqx.filter_vmap(Affine)(jnp.ones(3)),
        in_axes=eqx.if_array(0),
    ),
    "Reshape (unconditional)": Reshape(Affine(scale=jnp.arange(1, 5)), (2, 2)),
    "Reshape (conditional)": Reshape(
        MaskedAutoregressive(
            KEY,
            transformer=Affine(),
            dim=4,
            cond_dim=1,
            nn_width=3,
            nn_depth=1,
        ),
        shape=(1, 4, 1),
        cond_shape=(),
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
    """Tests the transform_and_log_det and inverse_and_log_det methods,
    by 1) checking invertibility and 2) comparing log dets to those obtained with
    automatic differentiation.
    """
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
    y, logdet = bijection.transform_and_log_det(x, cond)
    assert logdet == pytest.approx(auto_log_det, abs=1e-4)
    assert y == pytest.approx(bijection.transform(x, cond))

    try:
        x_reconstructed, logdetinv = bijection.inverse_and_log_det(y, cond)
        assert logdetinv == pytest.approx(-auto_log_det, abs=1e-4)
        assert x == pytest.approx(x_reconstructed, abs=1e-4)
        assert x == pytest.approx(bijection.inverse(y, cond), abs=1e-4)

    except NotImplementedError:
        pass


class _AstractTestBijection(AbstractBijection):
    shape: tuple[int, ...] = ()
    cond_shape: tuple[int, ...] | None = None

    def transform(self, x, condition=None):
        return x

    def transform_and_log_det(self, x, condition=None):
        return x, jnp.zeros(())

    @abstractmethod
    def inverse(self, y, condition=None):
        pass

    @abstractmethod
    def inverse_and_log_det(self, y, condition=None):
        return y, jnp.zeros(())


class _TestBijection(_AstractTestBijection):
    # Test bijection (with inheritance + method overide)
    def transform(self, x, condition=None):  # Check overiding
        return x

    def inverse(self, y, condition=None):
        return y

    def inverse_and_log_det(self, y, condition=None):
        return y, jnp.zeros(())


test_cases = {
    "wrong x shape": (
        [(), None],
        [jnp.ones(2), None],
        "Expected input shape",
    ),
    "wrong condition shape": (
        [(), (2,)],
        [1, jnp.ones(3)],
        "Expected condition.shape",
    ),
    "missing condition": (
        [(), (2,)],
        [1, None],
        "Expected condition to be provided.",
    ),
}


@pytest.mark.parametrize(
    ("args", "inputs", "match"),
    test_cases.values(),
    ids=test_cases.keys(),
)
def test_argcheck(args, inputs, match):
    bijection = _TestBijection(*args)

    with pytest.raises(ValueError, match=match):
        bijection.transform(*inputs)

    with pytest.raises(ValueError, match=match):
        bijection.inverse(*inputs)

    with pytest.raises(ValueError, match=match):
        bijection.transform_and_log_det(*inputs)

    with pytest.raises(ValueError, match=match):
        bijection.inverse_and_log_det(*inputs)
