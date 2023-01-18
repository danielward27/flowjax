"""
Premade versions of common flow architetctures from ``flowjax.flows``.
"""
# Note that here although we could chain arbitrary bijections using `Chain`, here,
# we generally opt to use `Scan`, which avoids excessive compilation
# when the flow layers share the same structure.

from typing import Callable, Optional

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
from jax import random
from jax.nn.initializers import glorot_uniform
from jax.random import KeyArray

from flowjax.bijections import (
    AdditiveLinearCondition,
    BlockAutoregressiveNetwork,
    Chain,
    Coupling,
    Flip,
    Invert,
    MaskedAutoregressive,
    Permute,
    Scan,
    TanhLinearTails,
    TriangularAffine,
    RationalQuadraticSpline,
    Bijection,
)
from flowjax.distributions import Distribution, Transformed


class CouplingFlow(
    Transformed
):  # TODO allow flows to work on higher dimensional inputs?
    flow_layers: int
    nn_width: int
    nn_depth: int
    permute_strategy: str

    def __init__(
        self,
        key: KeyArray,
        base_dist: Distribution,
        transformer: Bijection,
        cond_dim: int = None,
        flow_layers: int = 8,
        nn_width: int = 40,
        nn_depth: int = 2,
        nn_activation: Callable = jnn.relu,
        invert: bool = True,
    ):
        """Coupling flow (https://arxiv.org/abs/1605.08803).

        Args:
            key (KeyArray): Jax PRNGKey.
            base_dist (Distribution): Base distribution.
            transformer (Bijection): Bijection to be parameterised by conditioner.
            cond_dim (int, optional): Dimension of conditioning variables. Defaults to None.
            flow_layers (int, optional): Number of coupling layers. Defaults to 5.
            nn_width (int, optional): Conditioner hidden layer size. Defaults to 40.
            nn_depth (int, optional): Conditioner depth. Defaults to 2.
            nn_activation (int, optional): Conditioner activation function. Defaults to jnn.relu.
            invert: (bool, optional): Whether to invert the bijection. Broadly, True will prioritise a faster `inverse` methods, leading to faster `log_prob`, False will prioritise faster `transform` methods, leading to faster `sample`. Defaults to True
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
                d=dim // 2,
                D=dim,
                cond_dim=cond_dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                nn_activation=nn_activation,
            )

            if permute_strategy == "flip":
                return Chain([coupling, Flip()])
            elif permute_strategy == "random":
                perm = Permute(random.permutation(p_key, jnp.arange(dim)))
                return Chain([coupling, perm])
            else:
                return coupling

        keys = random.split(key, flow_layers)
        layers = eqx.filter_vmap(make_layer)(keys)
        bijection = Scan(layers)
        bijection = Invert(bijection) if invert else bijection

        self.nn_width = nn_width
        self.nn_depth = nn_depth
        self.flow_layers = flow_layers
        self.permute_strategy = permute_strategy
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)
        super().__init__(base_dist, bijection)


class MaskedAutoregressiveFlow(Transformed):
    flow_layers: int
    nn_width: int
    nn_depth: int
    permute_strategy: str

    def __init__(
        self,
        key: KeyArray,
        base_dist: Distribution,
        transformer: Bijection,
        cond_dim: int = None,
        flow_layers: int = 8,
        nn_width: int = 40,
        nn_depth: int = 2,
        nn_activation: Callable = jnn.relu,
        invert: bool = True,
    ):
        """Masked autoregressive flow (https://arxiv.org/abs/1606.04934,
        https://arxiv.org/abs/1705.07057v4). Parameterises a transformer with an
        autoregressive neural network.

        Args:
            key (KeyArray): Random seed.
            base_dist (Distribution): Base distribution.
            transformer (Bijection): Bijection parameterised by autoregressive network.
            cond_dim (int, optional): _description_. Defaults to 0.
            flow_layers (int, optional): Number of flow layers. Defaults to 5.
            nn_width (int, optional): Number of hidden layers in neural network. Defaults to 40.
            nn_depth (int, optional): Depth of neural network. Defaults to 2.
            nn_activation (Callable, optional): _description_. Defaults to jnn.relu.
            invert (bool, optional): Whether to invert the bijection. Broadly, True will
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
                return Chain([masked_autoregressive, Flip()])
            elif permute_strategy == "random":
                perm = Permute(random.permutation(p_key, jnp.arange(dim)))
                return Chain([masked_autoregressive, perm])
            else:
                return masked_autoregressive

        keys = random.split(key, flow_layers)
        layers = eqx.filter_vmap(make_layer)(keys)
        bijection = Scan(layers)
        bijection = Invert(bijection) if invert else bijection

        self.nn_width = nn_width
        self.nn_depth = nn_depth
        self.flow_layers = flow_layers
        self.permute_strategy = permute_strategy
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)
        super().__init__(base_dist, bijection)


class BlockNeuralAutoregressiveFlow(Transformed):
    flow_layers: int
    nn_block_dim: int
    nn_depth: int
    permute_strategy: str

    def __init__(
        self,
        key: KeyArray,
        base_dist: Distribution,
        cond_dim: Optional[int] = None,
        nn_depth: int = 1,
        nn_block_dim: int = 8,
        flow_layers: int = 1,
        invert: bool = True,
    ):
        """Block neural autoregressive flow (BNAF) (https://arxiv.org/abs/1904.04676).

        Args:
            key (KeyArray): Jax PRNGKey.
            base_dist (Distribution): Base distribution.
            cond_dim (Union[None, Tuple[int]]): Dimension of conditional variables.
            nn_depth (int, optional): Number of hidden layers within the networks. Defaults to 1.
            nn_block_dim (int, optional): Block size. Hidden layer width is dim*nn_block_dim. Defaults to 8.
            flow_layers (int, optional): Number of BNAF layers. Defaults to 1.
            invert: (bool, optional): Use `True` for access of `log_prob` only (e.g. fitting by maximum likelihood), `False` for the forward direction (sampling) only (e.g. for fitting variationally).
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
                return Chain([ban, Flip()])
            elif permute_strategy == "random":
                perm = Permute(random.permutation(p_key, jnp.arange(dim)))
                return Chain([ban, perm])
            else:
                return ban

        keys = random.split(key, flow_layers)
        layers = eqx.filter_vmap(make_layer)(keys)
        bijection = Scan(layers)
        bijection = Invert(bijection) if invert else bijection

        self.nn_block_dim = nn_block_dim
        self.nn_depth = nn_depth
        self.flow_layers = flow_layers
        self.permute_strategy = permute_strategy

        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)
        super().__init__(base_dist, bijection)


class TriangularSplineFlow(Transformed):
    flow_layers: int
    permute_strategy: str
    knots: int
    tanh_max_val: float

    def __init__(
        self,
        key: KeyArray,
        base_dist: Distribution,
        cond_dim: Optional[int] = None,
        flow_layers: int = 8,
        knots: int = 8,
        tanh_max_val: float = 3.0,
        invert: bool = True,
        permute_strategy: Optional[str] = None,
        init: Callable = glorot_uniform(),
    ):

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
            lt = TriangularAffine(jnp.zeros(dim), lt_weights, weight_normalisation=True)

            bijections = [
                TanhLinearTails(tanh_max_val),
                RationalQuadraticSpline(knots, interval=1, shape=(dim,)),
                Invert(TanhLinearTails(tanh_max_val)),
                lt,
            ]

            if cond_dim is not None:
                linear_condition = AdditiveLinearCondition(cond_weights)
                bijections.append(linear_condition)

            if permute_strategy == "flip":
                bijections.append(Flip())
            elif permute_strategy == "random":
                bijections.append(
                    Permute(random.permutation(perm_key, jnp.arange(dim)))
                )
            return Chain(bijections)

        keys = random.split(key, flow_layers)
        layers = eqx.filter_vmap(make_layer)(keys)
        bijection = Scan(layers)
        bijection = Invert(bijection) if invert else bijection

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
    else:
        return "random"
