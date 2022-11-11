"""Contains some common flow architectures. These act as convenience costructors,
and examples of how flows can be constructed."""

from typing import Optional
from jax import random
import jax.nn as jnn
import jax.numpy as jnp
from jax.random import KeyArray
from flowjax.bijections.utils import Permute, Flip
from flowjax.distributions import Distribution, Transformed
from typing import List
from flowjax.bijections import (
    Bijection, Transformer, Coupling, Chain, Invert,
    BlockAutoregressiveNetwork, MaskedAutoregressive
    )

def coupling_flow(
    key: KeyArray,
    base_dist: Distribution,
    transformer: Transformer,
    cond_dim: int = 0,
    flow_layers: int = 5,
    nn_width: int = 40,
    nn_depth: int = 2,
    permute_strategy: Optional[str] = None,
    nn_activation: int = jnn.relu,
    invert: bool = True
):
    """Coupling flow (https://arxiv.org/abs/1605.08803).

        Args:
            key (KeyArray): Jax PRNGKey.
            base_dist (Distribution): Base distribution.
            transformer (Transformer): Transformer parameterised by conditioner.
            cond_dim (int, optional): Dimension of conditioning variables. Defaults to 0.
            flow_layers (int, optional): Number of coupling layers. Defaults to 5.
            nn_width (int, optional): Conditioner hidden layer size. Defaults to 40.
            nn_depth (int, optional): Conditioner depth. Defaults to 2.
            permute_strategy (Optional[str], optional): "flip", "random" or "none". Defaults to "flip" if dim==2, and "random" for dim > 2.
            nn_activation (int, optional): Conditioner activation function. Defaults to jnn.relu.
            invert: (bool, optional): Whether to invert the bijection. Broadly, True will prioritise a faster `inverse` methods, leading to faster `log_prob`, False will prioritise faster `transform` methods, leading to faster `sample`. Defaults to True
    """

    if permute_strategy is None:
        permute_strategy = default_permute_strategy(base_dist.dim)
    if permute_strategy not in ["flip", "random", "none"]:
        raise ValueError("Permute strategy should be 'flip', 'random' or 'none', if specified.")

    bijections = [] # type: List[Bijection]
    for i in range(flow_layers):
        key, *subkeys = random.split(key, 3)
        bijections.append(
            Coupling(
                key=subkeys[0],
                transformer=transformer,
                d=base_dist.dim // 2,
                D=base_dist.dim,
                cond_dim=cond_dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                nn_activation=nn_activation
            )
        )
        if permute_strategy == "random" and i != flow_layers:
            perm = random.permutation(subkeys[1], jnp.arange(base_dist.dim))
            bijections.append(Permute(perm))
        elif permute_strategy == "flip" and i != flow_layers:
            bijections.append(Flip())
            
    bijection = Chain(bijections)
    if invert is True:
        bijection = Invert(bijection)
    return Transformed(base_dist, bijection)


def masked_autoregressive_flow(
    key: KeyArray,
    base_dist: Distribution,
    transformer: Transformer,
    cond_dim: int = 0,
    flow_layers: int = 5,
    nn_width: int = 40,
    nn_depth: int = 2,
    permute_strategy: Optional[str] = None,
    nn_activation: int = jnn.relu,
    invert: bool = True
    ):
    """Masked autoregressive flow (https://arxiv.org/abs/1705.07057v4). Parameterises a
    a transformer with a neural network with masking of weights to enforces the
    autoregressive property.

    Args:
        key (KeyArray): Random seed.
        base_dist (Distribution): Base distribution
        transformer (Transformer): Transformer parameterised by conditioner.
        nn_depth (int, optional): Depth of neural network. Defaults to 2.
        nn_width (int, optional): Number of hidden layers in neural network. Defaults to 60.
        flow_layers (int, optional): Number of `MaskedAutoregressive` layers. Defaults to 5.
        permute_strategy (Optional[str], optional): "flip", "random" or "none". Defaults to "flip" if dim==2, and "random" for dim > 2.
        invert: (bool, optional): Whether to invert the bijection. Broadly, True will prioritise a faster inverse, leading to faster `log_prob`, False will prioritise faster forward, leading to faster `sample`. Defaults to True
    """
    if permute_strategy is None:
        permute_strategy = default_permute_strategy(base_dist.dim)
    if permute_strategy not in ["flip", "random", "none"]:
        raise ValueError("Permute strategy should be 'flip', 'random' or 'none', if specified.")

    bijections = [] # type: List[Bijection]
    for i in range(flow_layers):
        key, *subkeys = random.split(key, 3)
        bijections.append(
            MaskedAutoregressive(
                key=subkeys[0],
                transformer=transformer,
                dim=base_dist.dim,
                cond_dim=cond_dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                nn_activation=nn_activation
            )
        )
        
        if permute_strategy == "random" and i != flow_layers:
            perm = random.permutation(subkeys[1], jnp.arange(base_dist.dim))
            bijections.append(Permute(perm))
        elif permute_strategy == "flip" and i != flow_layers:
            bijections.append(Flip())
    
    bijection = Chain(bijections)
    if invert is True:
        bijection = Invert(bijection)
    return Transformed(base_dist, bijection)

def block_neural_autoregressive_flow(
    key: KeyArray,
    base_dist: Distribution,
    cond_dim: int = 0,
    nn_depth: int = 1,
    nn_block_dim: int = 8,
    flow_layers: int = 1,
    permute_strategy: Optional[str] = None,
    invert: bool = True
):
    """Block neural autoregressive flow (BNAF) (https://arxiv.org/abs/1904.04676).

    Args:
        key (KeyArray): Jax PRNGKey.
        base_dist (Distribution): Base distribution.
        cond_dim (int): Dimension of conditional variables.
        nn_depth (int, optional): Number of hidden layers within the networks. Defaults to 1.
        nn_block_dim (int, optional): Block size. Hidden layer width is dim*nn_block_dim. Defaults to 8.
        flow_layers (int, optional): Number of BNAF layers. Defaults to 1.
        permute_strategy (Optional[str], optional): How to permute between layers. "flip", "random" or "none". Defaults to "flip" if dim==2, and "random" for dim > 2.
        invert: (bool, optional): Use `True` for access of `log_prob` only (e.g. fitting by maximum likelihood), `False` for the forward direction (sampling) only (e.g. for fitting variationally).
    """
    if permute_strategy is None:
        permute_strategy = default_permute_strategy(base_dist.dim)
    if permute_strategy not in ["flip", "random", "none"]:
        raise ValueError("Permute strategy should be 'flip', 'random' or 'none', if specified.")

    bijections = [] # type: List[Bijection]
    for i in range(flow_layers):
        key, *subkeys = random.split(key, 3)
        bijections.append(
            BlockAutoregressiveNetwork(
                key=subkeys[0],
                dim=base_dist.dim,
                cond_dim=cond_dim,
                depth=nn_depth,
                block_dim=nn_block_dim,
            )
        )

        if permute_strategy == "random" and i != flow_layers:
            perm = random.permutation(subkeys[1], jnp.arange(base_dist.dim))
            bijections.append(Permute(perm))
        elif permute_strategy == "flip" and i != flow_layers:
            bijections.append(Flip())

    bijection = Chain(bijections)
    if invert is True:
        bijection = Invert(bijection)
    return Transformed(base_dist, bijection)


def default_permute_strategy(dim):
    if dim <= 2:
        return {1: "none", 2: "flip"}[dim]
    else:
        return "random"
