"""Premade versions of common flow architetctures from ``flowjax.flows``.

All these functions return a :class:`~flowjax.distributions.Transformed` distribution.
"""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from equinox.nn import Linear
from jax.nn import softplus
from jax.nn.initializers import glorot_uniform
from jaxtyping import PRNGKeyArray

from flowjax.bijections import (
    AbstractBijection,
    AdditiveCondition,
    Affine,
    BlockAutoregressiveNetwork,
    Chain,
    Coupling,
    Flip,
    Invert,
    LeakyTanh,
    MaskedAutoregressive,
    Permute,
    Planar,
    RationalQuadraticSpline,
    Scan,
    TriangularAffine,
    Vmap,
)
from flowjax.distributions import AbstractDistribution, Transformed
from flowjax.utils import inv_softplus
from flowjax.wrappers import Parameterize, WeightNormalization


def _affine_with_min_scale(min_scale: float = 1e-2) -> Affine:
    scale = Parameterize(lambda x: softplus(x) + min_scale, inv_softplus(1 - min_scale))
    return eqx.tree_at(where=lambda aff: aff.scale, pytree=Affine(), replace=scale)


def coupling_flow(
    key: PRNGKeyArray,
    *,
    base_dist: AbstractDistribution,
    transformer: AbstractBijection | None = None,
    cond_dim: int | None = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = jnn.relu,
    invert: bool = True,
) -> Transformed:
    """Create a coupling flow (https://arxiv.org/abs/1605.08803).

    Args:
        key: Jax random key.
        base_dist: Base distribution, with ``base_dist.ndim==1``.
        transformer: Bijection to be parameterised by conditioner. Defaults to
            affine.
        cond_dim: Dimension of conditioning variables. Defaults to None.
        flow_layers: Number of coupling layers. Defaults to 8.
        nn_width: Conditioner hidden layer size. Defaults to 50.
        nn_depth: Conditioner depth. Defaults to 1.
        nn_activation: Conditioner activation function. Defaults to jnn.relu.
        invert: Whether to invert the bijection. Broadly, True will prioritise a faster
            `inverse` methods, leading to faster `log_prob`, False will prioritise
            faster `transform` methods, leading to faster `sample`. Defaults to True.
    """
    if transformer is None:
        transformer = _affine_with_min_scale()

    dim = base_dist.shape[-1]

    def make_layer(key):  # coupling layer + permutation
        bij_key, perm_key = jr.split(key)
        bijection = Coupling(
            key=bij_key,
            transformer=transformer,
            untransformed_dim=dim // 2,
            dim=dim,
            cond_dim=cond_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
        )
        return _add_default_permute(bijection, dim, perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


def masked_autoregressive_flow(
    key: PRNGKeyArray,
    *,
    base_dist: AbstractDistribution,
    transformer: AbstractBijection | None = None,
    cond_dim: int | None = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = jnn.relu,
    invert: bool = True,
) -> Transformed:
    """Masked autoregressive flow.

    Parameterises a transformer bijection with an autoregressive neural network.
    Refs: https://arxiv.org/abs/1606.04934; https://arxiv.org/abs/1705.07057v4.

    Args:
        key: Random seed.
        base_dist: Base distribution, with ``base_dist.ndim==1``.
        transformer: Bijection parameterised by autoregressive network. Defaults to
            affine.
        cond_dim: Dimension of the conditioning variable. Defaults to None.
        flow_layers: Number of flow layers. Defaults to 8.
        nn_width: Number of hidden layers in neural network. Defaults to 50.
        nn_depth: Depth of neural network. Defaults to 1.
        nn_activation: _description_. Defaults to jnn.relu.
        invert: Whether to invert the bijection. Broadly, True will prioritise a faster
            inverse, leading to faster `log_prob`, False will prioritise faster forward,
            leading to faster `sample`. Defaults to True.
    """
    if transformer is None:
        transformer = _affine_with_min_scale()

    dim = base_dist.shape[-1]

    def make_layer(key):  # masked autoregressive layer + permutation
        bij_key, perm_key = jr.split(key)
        bijection = MaskedAutoregressive(
            key=bij_key,
            transformer=transformer,
            dim=dim,
            cond_dim=cond_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
        )
        return _add_default_permute(bijection, dim, perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


def block_neural_autoregressive_flow(
    key: PRNGKeyArray,
    *,
    base_dist: AbstractDistribution,
    cond_dim: int | None = None,
    nn_depth: int = 1,
    nn_block_dim: int = 8,
    flow_layers: int = 1,
    invert: bool = True,
    activation: AbstractBijection | Callable | None = None,
    inverter: Callable | None = None,
) -> Transformed:
    """Block neural autoregressive flow (BNAF) (https://arxiv.org/abs/1904.04676).

    Each flow layer contains a
    :class:`~flowjax.bijections.block_autoregressive_network.BlockAutoregressiveNetwork`
    bijection. The bijection does not have an analytic inverse, so must be inverted
    using numerical methods (by default a bisection search). Note that this means
    that only one of ``log_prob`` or ``sample{_and_log_prob}`` can be efficient,
    controlled by the ``invert`` argument.

    Args:
        key: Jax key.
        base_dist: Base distribution, with ``base_dist.ndim==1``.
        cond_dim: Dimension of conditional variables. Defaults to None.
        nn_depth: Number of hidden layers within the networks. Defaults to 1.
        nn_block_dim: Block size. Hidden layer width is dim*nn_block_dim. Defaults to 8.
        flow_layers: Number of BNAF layers. Defaults to 1.
        invert: Use `True` for efficient ``log_prob`` (e.g. when fitting by maximum
            likelihood), and `False` for efficient ``sample`` and
            ``sample_and_log_prob`` methods (e.g. for fitting variationally).
        activation: Activation function used within block neural autoregressive
            networks. Note this should be bijective and in some use cases should map
            real -> real. For more information, see
            :class:`~flowjax.bijections.block_autoregressive_network.BlockAutoregressiveNetwork`.
            Defaults to :class:`~flowjax.bijections.tanh.LeakyTanh`.
        inverter: Callable that implements the required numerical method to invert the
            ``BlockAutoregressiveNetwork`` bijection. Must have the signature
            ``inverter(bijection, y, condition=None)``. Defaults to using a bisection
            search via ``AutoregressiveBisectionInverter``.
    """

    def make_layer(key):  # bnaf layer + permutation
        bij_key, perm_key = jr.split(key)
        bijection = BlockAutoregressiveNetwork(
            bij_key,
            dim=base_dist.shape[-1],
            cond_dim=cond_dim,
            depth=nn_depth,
            block_dim=nn_block_dim,
            activation=activation,
            inverter=inverter,
        )
        return _add_default_permute(bijection, base_dist.shape[-1], perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


def planar_flow(
    key: PRNGKeyArray,
    *,
    base_dist: AbstractDistribution,
    cond_dim: int | None = None,
    flow_layers: int = 8,
    invert: bool = True,
    negative_slope: float | None = None,
    **mlp_kwargs,
) -> Transformed:
    """Planar flow as introduced in https://arxiv.org/pdf/1505.05770.pdf.

    This alternates between :class:`~flowjax.bijections.planar.Planar` layers and
    permutations. Note the definition here is inverted compared to the original paper.

    Args:
        key: Jax key.
        base_dist: Base distribution, with ``base_dist.ndim==1``.
        cond_dim: Dimension of conditioning variables. Defaults to None.
        flow_layers: Number of flow layers. Defaults to 8.
        invert: Whether to invert the bijection. Broadly, True will prioritise a faster
            `inverse` methods, leading to faster `log_prob`, False will prioritise
            faster `transform` methods, leading to faster `sample`. Defaults to True.
        negative_slope: A positive float. If provided, then a leaky relu activation
            (with the corresponding negative slope) is used instead of tanh. This also
            provides the advantage that the bijection can be inverted analytically.
        **mlp_kwargs: Keyword arguments (excluding in_size and out_size) passed to
            the MLP (equinox.nn.MLP). Ignored when cond_dim is None.
    """

    def make_layer(key):  # Planar layer + permutation
        bij_key, perm_key = jr.split(key)
        bijection = Planar(
            bij_key,
            dim=base_dist.shape[-1],
            cond_dim=cond_dim,
            negative_slope=negative_slope,
            **mlp_kwargs,
        )
        return _add_default_permute(bijection, base_dist.shape[-1], perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


def triangular_spline_flow(
    key: PRNGKeyArray,
    *,
    base_dist: AbstractDistribution,
    cond_dim: int | None = None,
    flow_layers: int = 8,
    knots: int = 8,
    tanh_max_val: float | int = 3.0,
    invert: bool = True,
    init: Callable | None = None,
) -> Transformed:
    """Triangular spline flow.

    A single layer consists where each layer consists of a triangular affine
    transformation with weight normalisation, and an elementwise rational quadratic
    spline. Tanh is used to constrain to the input to [-1, 1] before spline
    transformations.

    Args:
        key: Jax random key.
        base_dist: Base distribution, with ``base_dist.ndim==1``.
        cond_dim: The number of conditioning features. Defaults to None.
        flow_layers: Number of flow layers. Defaults to 8.
        knots: Number of knots in the splines. Defaults to 8.
        tanh_max_val: Maximum absolute value beyond which we use linear "tails" in the
            tanh function. Defaults to 3.0.
        invert: Whether to invert the bijection before transforming the base
            distribution. Defaults to True.
        init: Initialisation method for the lower triangular weights.
            Defaults to glorot_uniform().
    """
    init = init if init is not None else glorot_uniform()
    dim = base_dist.shape[-1]

    def get_splines():
        fn = partial(RationalQuadraticSpline, knots=knots, interval=1)
        spline = eqx.filter_vmap(fn, axis_size=dim)()
        return Vmap(spline, in_axes=eqx.if_array(0))

    def make_layer(key):
        lt_key, perm_key, cond_key = jr.split(key, 3)
        weights = init(lt_key, (dim, dim))
        lt_weights = weights.at[jnp.diag_indices(dim)].set(1)
        tri_aff = TriangularAffine(jnp.zeros(dim), lt_weights)
        tri_aff = eqx.tree_at(
            lambda t: t.triangular, tri_aff, replace_fn=WeightNormalization
        )
        bijections = [
            LeakyTanh(tanh_max_val, (dim,)),
            get_splines(),
            Invert(LeakyTanh(tanh_max_val, (dim,))),
            tri_aff,
        ]

        if cond_dim is not None:
            linear_condition = AdditiveCondition(
                Linear(cond_dim, dim, use_bias=False, key=cond_key),
                (dim,),
                (cond_dim,),
            )
            bijections.append(linear_condition)

        bijection = Chain(bijections)
        return _add_default_permute(bijection, dim, perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


def _add_default_permute(bijection: AbstractBijection, dim: int, key: PRNGKeyArray):
    if dim == 1:
        return bijection
    if dim == 2:
        return Chain([bijection, Flip((dim,))]).merge_chains()

    perm = Permute(jr.permutation(key, jnp.arange(dim)))
    return Chain([bijection, perm]).merge_chains()
