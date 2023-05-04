"""Premade versions of common flow architetctures from ``flowjax.flows``."""
# Note that here although we could chain arbitrary bijections using `Chain`, here,
# we generally opt to use `Scan`, which avoids excessive compilation
# when the flow layers share the same structure.

from typing import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
from jax import random
from jax.nn.initializers import glorot_uniform
from jax.random import KeyArray

from flowjax.bijections import (
    AdditiveLinearCondition,
    Bijection,
    BlockAutoregressiveNetwork,
    Chain,
    Coupling,
    Flip,
    Invert,
    MaskedAutoregressive,
    Permute,
    RationalQuadraticSpline,
    Scan,
    TanhLinearTails,
    TriangularAffine,
)
from flowjax.distributions import Distribution, Transformed


class CouplingFlow(Transformed):
    """Coupling flow (https://arxiv.org/abs/1605.08803)."""

    flow_layers: int
    nn_width: int
    nn_depth: int
    permute_strategy: str

    def __init__(
        self,
        key: KeyArray,
        base_dist: Distribution,
        transformer: Bijection,
        cond_dim: int | None = None,
        flow_layers: int = 8,
        nn_width: int = 40,
        nn_depth: int = 2,
        nn_activation: Callable = jnn.relu,
        invert: bool = True,
    ):
        """
        Args:
            key (KeyArray): Jax PRNGKey.
            base_dist (Distribution): Base distribution.
            transformer (Bijection): Bijection to be parameterised by conditioner.
            cond_dim (int): Dimension of conditioning variables. Defaults to None.
            flow_layers (int): Number of coupling layers. Defaults to 5.
            nn_width (int): Conditioner hidden layer size. Defaults to 40.
            nn_depth (int): Conditioner depth. Defaults to 2.
            nn_activation (int): Conditioner activation function. Defaults to jnn.relu.
            invert: (bool): Whether to invert the bijection. Broadly, True will
                prioritise a faster `inverse` methods, leading to faster `log_prob`,
                False will prioritise faster `transform` methods, leading to faster
                `sample`. Defaults to True
        """
        if base_dist.ndim != 1:
            raise ValueError(f"Expected base_dist.ndim==1, got {base_dist.ndim}")

        dim = base_dist.shape[0]
        permute_strategy = _default_permute_strategy(dim)

        def make_layer(key):  # coupling layer + permutation
            c_key, p_key = random.split(key)
            coupling = Coupling(
                key=c_key,
                transformer=transformer,
                untransformed_dim=dim // 2,
                dim=dim,
                cond_dim=cond_dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                nn_activation=nn_activation,
            )

            if permute_strategy == "flip":
                return Chain([coupling, Flip((dim,))])
            if permute_strategy == "random":
                perm = Permute(random.permutation(p_key, jnp.arange(dim)))
                return Chain([coupling, perm])
            return coupling

        keys = random.split(key, flow_layers)
        layers = eqx.filter_vmap(make_layer)(keys)
        bijection = Invert(Scan(layers)) if invert else Scan(layers)

        self.nn_width = nn_width
        self.nn_depth = nn_depth
        self.flow_layers = flow_layers
        self.permute_strategy = permute_strategy
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)
        super().__init__(base_dist, bijection)


class MaskedAutoregressiveFlow(Transformed):
    """Masked autoregressive flow (https://arxiv.org/abs/1606.04934,
    https://arxiv.org/abs/1705.07057v4). Parameterises a transformer with an
    autoregressive neural network.
    """

    flow_layers: int
    nn_width: int
    nn_depth: int
    permute_strategy: str

    def __init__(
        self,
        key: KeyArray,
        base_dist: Distribution,
        transformer: Bijection,
        cond_dim: int | None = None,
        flow_layers: int = 8,
        nn_width: int = 40,
        nn_depth: int = 2,
        nn_activation: Callable = jnn.relu,
        invert: bool = True,
    ):
        """
        Args:
            key (KeyArray): Random seed.
            base_dist (Distribution): Base distribution.
            transformer (Bijection): Bijection parameterised by autoregressive network.
            cond_dim (int): _description_. Defaults to 0.
            flow_layers (int): Number of flow layers. Defaults to 5.
            nn_width (int): Number of hidden layers in neural network. Defaults to 40.
            nn_depth (int): Depth of neural network. Defaults to 2.
            nn_activation (Callable): _description_. Defaults to jnn.relu.
            invert (bool): Whether to invert the bijection. Broadly, True will
                prioritise a faster inverse, leading to faster `log_prob`, False will prioritise
                faster forward, leading to faster `sample`. Defaults to True. Defaults to True.
        """
        if len(base_dist.shape) != 1:
            raise ValueError(f"Expected base_dist.ndim==1, got {base_dist.ndim}")

        dim = base_dist.shape[0]
        permute_strategy = _default_permute_strategy(dim)

        def make_layer(key):  # masked autoregressive layer + permutation
            masked_auto_key, p_key = random.split(key)
            masked_autoregressive = MaskedAutoregressive(
                key=masked_auto_key,
                transformer=transformer,
                dim=dim,
                cond_dim=cond_dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                nn_activation=nn_activation,
            )
            if permute_strategy == "flip":
                return Chain([masked_autoregressive, Flip((dim,))])
            if permute_strategy == "random":
                perm = Permute(random.permutation(p_key, jnp.arange(dim)))
                return Chain([masked_autoregressive, perm])
            return masked_autoregressive

        keys = random.split(key, flow_layers)
        layers = eqx.filter_vmap(make_layer)(keys)
        bijection = Invert(Scan(layers)) if invert else Scan(layers)

        self.nn_width = nn_width
        self.nn_depth = nn_depth
        self.flow_layers = flow_layers
        self.permute_strategy = permute_strategy
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)
        super().__init__(base_dist, bijection)


class BlockNeuralAutoregressiveFlow(Transformed):
    """Block neural autoregressive flow (BNAF) (https://arxiv.org/abs/1904.04676)."""

    flow_layers: int
    nn_block_dim: int
    nn_depth: int
    permute_strategy: str

    def __init__(
        self,
        key: KeyArray,
        base_dist: Distribution,
        cond_dim: int | None = None,
        nn_depth: int = 1,
        nn_block_dim: int = 8,
        flow_layers: int = 1,
        invert: bool = True,
    ):
        """
        Args:
            key (KeyArray): Jax PRNGKey.
            base_dist (Distribution): Base distribution.
            cond_dim (int | None): Dimension of conditional variables.
            nn_depth (int): Number of hidden layers within the networks.
                Defaults to 1.
            nn_block_dim (int): Block size. Hidden layer width is
                dim*nn_block_dim. Defaults to 8.
            flow_layers (int): Number of BNAF layers. Defaults to 1.
            invert: (bool): Use `True` for access of `log_prob` only (e.g.
                fitting by maximum likelihood), `False` for the forward direction
                (sampling) only (e.g. for fitting variationally).
        """
        if len(base_dist.shape) != 1:
            raise ValueError(f"Expected base_dist.ndim==1, got {base_dist.ndim}")

        dim = base_dist.shape[-1]
        permute_strategy = _default_permute_strategy(base_dist.shape[0])

        def make_layer(key):  # masked autoregressive layer + permutation
            ban_key, p_key = random.split(key)
            ban = BlockAutoregressiveNetwork(
                key=ban_key,
                dim=dim,
                cond_dim=cond_dim,
                depth=nn_depth,
                block_dim=nn_block_dim,
            )
            if permute_strategy == "flip":
                return Chain([ban, Flip((dim,))])
            if permute_strategy == "random":
                perm = Permute(random.permutation(p_key, jnp.arange(dim)))
                return Chain([ban, perm])
            return ban

        keys = random.split(key, flow_layers)
        layers = eqx.filter_vmap(make_layer)(keys)
        bijection = Invert(Scan(layers)) if invert else Scan(layers)

        self.nn_block_dim = nn_block_dim
        self.nn_depth = nn_depth
        self.flow_layers = flow_layers
        self.permute_strategy = permute_strategy

        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)
        super().__init__(base_dist, bijection)


class TriangularSplineFlow(Transformed):
    """A stack of layers, where each layer consists of a triangular affine
    transformation with weight normalisation, and an elementwise rational quadratic
    spline. Tanh is used to constrain to the input to [-1, 1] before spline
    transformations.
    """

    flow_layers: int
    permute_strategy: str
    knots: int
    tanh_max_val: float

    def __init__(
        self,
        key: KeyArray,
        base_dist: Distribution,
        cond_dim: int | None = None,
        flow_layers: int = 8,
        knots: int = 8,
        tanh_max_val: float = 3.0,
        invert: bool = True,
        init: Callable = glorot_uniform(),
    ):
        """
        Args:
            key (KeyArray): Jax random seed.
            base_dist (Distribution): Base distribution of the flow.
            cond_dim (int | None): The number of conditioning features.
                Defaults to None.
            flow_layers (int): Number of flow layers. Defaults to 8.
            knots (int): Number of knots in the splines. Defaults to 8.
            tanh_max_val (float): Maximum absolute value beyond which we use linear
                "tails" in the tanh function. Defaults to 3.0.
            invert: (bool): Use `True` for access of `log_prob` only (e.g.
                fitting by maximum likelihood), `False` for the forward direction
                (sampling) only (e.g. for fitting variationally).
            init (Callable): Initialisation method for the lower triangular weights.
                Defaults to glorot_uniform().
        """
        if len(base_dist.shape) != 1:
            raise ValueError(f"Expected base_dist.ndim==1, got {base_dist.ndim}")

        dim = base_dist.shape[-1]
        permute_strategy = _default_permute_strategy(dim)

        def make_layer(key):
            lt_key, perm_key = random.split(key)
            c_dim = 0 if cond_dim is None else cond_dim
            weights = init(lt_key, (dim, dim + c_dim))
            lt_weights = weights[:, :dim].at[jnp.diag_indices(dim)].set(1)
            cond_weights = weights[:, dim:]
            lower_tri = TriangularAffine(
                jnp.zeros(dim), lt_weights, weight_normalisation=True
            )

            bijections = [
                TanhLinearTails(tanh_max_val, (dim,)),
                RationalQuadraticSpline(knots, interval=1, shape=(dim,)),
                Invert(TanhLinearTails(tanh_max_val, (dim,))),
                lower_tri,
            ]

            if cond_dim is not None:
                linear_condition = AdditiveLinearCondition(cond_weights)
                bijections.append(linear_condition)

            if permute_strategy == "flip":
                bijections.append(Flip((dim,)))
            elif permute_strategy == "random":
                bijections.append(
                    Permute(random.permutation(perm_key, jnp.arange(dim)))
                )
            return Chain(bijections)

        keys = random.split(key, flow_layers)
        layers = eqx.filter_vmap(make_layer)(keys)
        bijection = Invert(Scan(layers)) if invert else Scan(layers)

        self.flow_layers = flow_layers
        self.permute_strategy = permute_strategy
        self.knots = knots
        self.tanh_max_val = tanh_max_val
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)

        super().__init__(base_dist, bijection)


def _default_permute_strategy(dim):
    if dim <= 2:
        return {1: "none", 2: "flip"}[dim]
    return "random"
