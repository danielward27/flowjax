from typing import Callable, Optional
import jax.numpy as jnp
from flowjax.bijections.abc import Bijection, ParameterisedBijection
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
        bijection: Bijection,  # TODO can/should the bijection can specify the cond dim?
    ):
        """Form a distribution like object using a base distribution and a
        bijection. Operations are generally assumed to be batched along
        dimension 0. dim is ignored if a base distribution is specified.
        Either dim or base_dist should be specified. If dim is specified,
        the base distribution

        Args:
            bijection (Bijection): Bijection mapping from target distribution to
                the base distribution.
            dim: (int): Dimension of the target distribution.
            base_log_prob (Optional[Callable], optional): log probability in the base
                distribution. Defaults to standard normal.
            base_sample (Optional[Callable], optional): sample function with signature
                (key : PRNGKey, n : int). Defaults to standard normal.
        """  # TODO update these docs
        self.base_dist = base_dist
        self.bijection = bijection
        self.dim = self.base_dist.dim
        self.cond_dim = max(
            self.bijection.cond_dim, self.base_dist.cond_dim
        )  # TODO bit odd, but either could be conditional or unconditional...

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        "Evaluate the log probability of the target distribution."
        z, log_abs_det = self.bijection.transform_and_log_abs_det_jacobian(x, condition)
        p_z = self.base_dist._log_prob(z, condition)
        return p_z + log_abs_det

    def _sample(
        self, key: KeyArray, condition: Optional[Array] = None,
    ):
        """Sample from the (conditional or unconditional) flow. For repeated sampling using
        a particular instance of the conditioning variable, use a vector condition and n to
        specify the number of samples. To sample once for may different conditioning variables
        provide a matrix of conditioning variables (n is inferred from axis 0).

        Args:
            key KeyArray: Random key (jax.random.PRNGKey).
            condition (Array, optional): Conditioning variables. Defaults to None.
            n (Optional[int], optional): Number of samples. Defaults to None.

        Returns:
            Array: Samples from the target distribution.
        """  # TODO update these docs
        z = self.base_dist._sample(key, condition)
        x = self.bijection.inverse(z, condition)
        return x


class CouplingFlow(Flow):
    def __init__(
        self,
        key: KeyArray,
        base_dist: Distribution,
        bijection: ParameterisedBijection,
        cond_dim: int = 0,
        num_layers: int = 5,
        nn_width: int = 40,
        nn_depth: int = 2,
        permute_strategy: Optional[str] = None,
    ):
        """Creates a flow with multiple Coupling Layers, with permutations inbetween.
        A RealNVP-style flow (Dinh et al, 2017; https://arxiv.org/abs/1605.08803) can
        be created by passing an `Affine` bijection instance. A neural spline flow
        (Durkan et al. 2019; https://arxiv.org/abs/1906.04032) can be created by
        passing a `RationalQuadraticSpline` instance. Note that the same bijection
        instance is used throughout, so if they contain trainable attributes, these
        will be shared (this is not an issue for bijections parameterised solely by
        neural networks).

        Args:
            key (KeyArray): Jax PRNGKey.
            base_dist (Distribution): Base distribution.
            bijection (ParameterisedBijection): Bijection parameterised by neural network.
            cond_dim (int, optional): Dimension of extra variables to condition on. Defaults to 0.
            num_layers (int, optional): Flow coupling layers. Defaults to 5.
            nn_width (int, optional): Conditioner hidden layer size. Defaults to 40.
            nn_depth (int, optional): Conditioner depth. Defaults to 2.
            permute_strategy (Optional[str], optional): "flip" or "random". Defaults to "flip" for 2 dimensional distributions, otherwise "random".
        """

        permute_key, *layer_keys = random.split(key, num_layers + 1)
        layers = [
            Coupling(
                key=key,
                bijection=bijection,
                d=base_dist.dim // 2,
                D=base_dist.dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                cond_dim=cond_dim,
            )
            for key in layer_keys
        ]  # type: List[Bijection]
        layers = intertwine_permute(
            permute_key, layers, base_dist.dim, permute_strategy,
        )
        super().__init__(base_dist, Chain(layers))


class BlockNeuralAutoregressiveFlow(Flow):
    def __init__(
        self,
        key: KeyArray,
        base_dist: Distribution,
        cond_dim: int = 0,
        nn_layers: int = 3,
        block_dim: int = 8,
        flow_layers: int = 1,
        permute_strategy: Optional[str] = None,
    ):
        """Convenience constructor for a block neural autoregressive flow
        (https://arxiv.org/abs/1904.04676).

        Args:
            key KeyArray: Random key.
            dim (int): Dimension of the target distribution.
            nn_layers (int, optional): Number of layers within autoregressive neural networks. Defaults to 3.
            block_dim (int, optional): Block size in lower triangular blocks of autoregressive neural network. Defaults to 8.
            flow_layers (int, optional): Number of flow layers (1 layer = autoregressive neural network + TanH activation) . Defaults to 1.
            permute_strategy (Optional[str], optional): How to permute between layers. Either "flip" or "random". Defaults to "flip" if dim <=2, otherwise "random".
            base_log_prob (Optional[Callable], optional): Base distribution log probability function. Defaults to standard normal.
            base_sample (Optional[Callable], optional): Base distribution sample function. Defaults to standard normal.
        """
        assert nn_layers >= 2

        permute_key, *layer_keys = random.split(key, flow_layers + 1)

        bijections = [
            BlockAutoregressiveNetwork(
                key,
                dim=base_dist.dim,
                cond_dim=cond_dim,
                n_layers=nn_layers,
                block_dim=block_dim,
            )
            for key in layer_keys
        ]  # type: List[Bijection]

        bijections = intertwine_permute(
            permute_key, bijections, base_dist.dim, permute_strategy,
        )
        bijection = Chain(bijections)
        super().__init__(base_dist, bijection)


class MaskedAutoregressiveFlow(Flow):
    def __init__(
        self,
        key: KeyArray,
        base_dist: Distribution,
        bijection: ParameterisedBijection,
        nn_depth: int = 2,
        nn_width: int = 60,
        nn_activation: Callable = jnn.relu,
        flow_layers: int = 5,
        permute_strategy: Optional[str] = None,
    ):
        """It is reccomended if feasible that the hidden dimension is at least
        the distribution dimension, to ensure all ranks can be represented.

        Args:
            key (KeyArray): Random seed.
            base_dist (Distribution): Base distribution
            nn_depth (int, optional): Depth of autoregressive neural network. Defaults to 2.
            nn_width (int, optional): _description_. Defaults to 60.
            nn_activation (Callable, optional): _description_. Defaults to jnn.relu.
            flow_layers (int, optional): _description_. Defaults to 5.
            permute_strategy (Optional[str], optional): "flip" or "random". Defaults to None.
        """
        # TODO Support conditional MAFs, and implement the inverse

        permute_key, *layer_keys = random.split(key, flow_layers + 1)

        bijections = [
            MaskedAutoregressive(
                key, bijection, base_dist.dim, nn_width, nn_depth, nn_activation
            )
            for key in layer_keys
        ]

        bijections = intertwine_permute(
            permute_key, bijections, base_dist.dim, permute_strategy,
        )
        bijection = Chain(bijections)
        super().__init__(base_dist, bijection)
