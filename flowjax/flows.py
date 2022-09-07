from typing import Optional
from flowjax.bijections.abc import Bijection, Transformer
from jax import random
import jax.nn as jnn
import equinox as eqx
from jax.random import KeyArray
from flowjax.bijections.coupling import Coupling
from flowjax.bijections.utils import Chain, intertwine_permute
from flowjax.bijections.bnaf import BlockAutoregressiveNetwork
from flowjax.bijections.masked_autoregressive import MaskedAutoregressive
from flowjax.distributions import Distribution
from flowjax.utils import Array
from typing import List

class Flow(eqx.Module, Distribution):
    bijection: Bijection
    base_dist: Distribution
    dim: int
    cond_dim: int

    def __init__(
        self,
        base_dist: Distribution,
        bijection: Bijection,
    ):
        """Form a distribution like object using a base distribution and a
        bijection. The bijection is defined in the "normalising" direction.

        Args:
            base_dist (Distribution): Base distribution.
            bijection (Bijection): Bijection defined in "normalising" direction.
        """
        self.base_dist = base_dist
        self.bijection = bijection
        self.dim = self.base_dist.dim
        self.cond_dim = max(self.bijection.cond_dim, self.base_dist.cond_dim)

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        z, log_abs_det = self.bijection.transform_and_log_abs_det_jacobian(x, condition)
        p_z = self.base_dist._log_prob(z, condition)
        return p_z + log_abs_det

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        z = self.base_dist._sample(key, condition)
        x = self.bijection.inverse(z, condition)
        return x


class CouplingFlow(Flow):
    def __init__(
        self,
        key: KeyArray,
        base_dist: Distribution,
        transformer: Transformer,
        cond_dim: int = 0,
        flow_layers: int = 5,
        nn_width: int = 40,
        nn_depth: int = 2,
        permute_strategy: Optional[str] = None,
        nn_activation: int = jnn.relu
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
            permute_strategy (Optional[str], optional): "flip" or "random". Defaults to "flip" for 2 dimensional distributions, otherwise "random".
            nn_activation (int, optional): Conditioner activation function. Defaults to jnn.relu.
        """

        permute_key, *layer_keys = random.split(key, flow_layers + 1)
        layers = [
            Coupling(
                key=key,
                transformer=transformer,
                d=base_dist.dim // 2,
                D=base_dist.dim,
                cond_dim=cond_dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                nn_activation=nn_activation
            )
            for key in layer_keys
        ]  # type: List[Bijection]
        layers = intertwine_permute(
            permute_key, layers, base_dist.dim, permute_strategy,
        )
        super().__init__(base_dist, Chain(layers))


class MaskedAutoregressiveFlow(Flow):
    def __init__(
        self,
        key: KeyArray,
        base_dist: Distribution,
        transformer: Transformer,
        cond_dim: int = 0,
        flow_layers: int = 5,
        nn_width: int = 40,
        nn_depth: int = 2,
        permute_strategy: Optional[str] = None,
        nn_activation: int = jnn.relu
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
            permute_strategy (Optional[str], optional): "flip" or "random". Defaults to None.
        """
        permute_key, *layer_keys = random.split(key, flow_layers + 1)

        bijections = [
            MaskedAutoregressive(
                key, transformer, base_dist.dim, cond_dim, nn_width, nn_depth, nn_activation
            )
            for key in layer_keys
        ]

        bijections = intertwine_permute(
            permute_key, bijections, base_dist.dim, permute_strategy,
        )
        transformer = Chain(bijections)
        super().__init__(base_dist, transformer)



class BlockNeuralAutoregressiveFlow(Flow):
    def __init__(
        self,
        key: KeyArray,
        base_dist: Distribution,
        cond_dim: int = 0,
        nn_depth: int = 1,
        nn_block_dim: int = 8,
        flow_layers: int = 1,
        permute_strategy: Optional[str] = None,
    ):
        """Block neural autoregressive flow (BNAF) (https://arxiv.org/abs/1904.04676).

        Args:
            key (KeyArray): Jax PRNGKey.
            base_dist (Distribution): Base distribution.
            cond_dim (int): Dimension of conditional variables.
            nn_depth (int, optional): Number of hidden layers within the networks. Defaults to 1.
            nn_block_dim (int, optional): Block size. Hidden layer width is dim*nn_block_dim. Defaults to 8.
            flow_layers (int, optional): Number of BNAF layers. Defaults to 1.
            permute_strategy (Optional[str], optional): How to permute between layers. Either "flip" or "random". Defaults to "flip" if dim==2, otherwise "random".
        """
        permute_key, *layer_keys = random.split(key, flow_layers + 1)

        bijections = [
            BlockAutoregressiveNetwork(
                key,
                dim=base_dist.dim,
                cond_dim=cond_dim,
                depth=nn_depth,
                block_dim=nn_block_dim,
            )
            for key in layer_keys
        ]  # type: List[Bijection]

        bijections = intertwine_permute(
            permute_key, bijections, base_dist.dim, permute_strategy,
        )
        bijection = Chain(bijections)
        super().__init__(base_dist, bijection)

