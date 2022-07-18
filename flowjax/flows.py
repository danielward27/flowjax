from typing import Callable, Optional
import jax.numpy as jnp
from flowjax.bijections.abc import Bijection
from jax import random
import equinox as eqx
import jax
from flowjax.bijections.coupling import Coupling
from flowjax.bijections.rational_quadratic_spline import RationalQuadraticSpline
from flowjax.bijections.affine import Affine
from flowjax.bijections.utils import Chain
from flowjax.bijections.bnaf import BlockAutoregressiveNetwork
from flowjax.bijections.utils import intertwine_permute
from flowjax.distributions import Distribution


class Flow(eqx.Module, Distribution):
    bijection: Bijection
    base_dist: Distribution
    dim: int
    cond_dim: int

    def __init__(
        self,
        base_dist: Distribution,
        bijection: Bijection,  # TODO can the bijection can specify the cond dim?
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

    def _log_prob(self, x: jnp.ndarray, condition: Optional[jnp.ndarray] = None):
        "Evaluate the log probability of the target distribution."
        z, log_abs_det = self.bijection.transform_and_log_abs_det_jacobian(x, condition)
        p_z = self.base_dist._log_prob(z, condition)
        return p_z + log_abs_det

    def _sample(
        self, key: random.PRNGKey, condition: Optional[jnp.ndarray] = None,
    ):
        """Sample from the (conditional or unconditional) flow. For repeated sampling using
        a particular instance of the conditioning variable, use a vector condition and n to
        specify the number of samples. To sample once for may different conditioning variables
        provide a matrix of conditioning variables (n is inferred from axis 0).

        Args:
            key (random.PRNGKey): Random key.
            condition (jnp.ndarray, optional): Conditioning variables. Defaults to None.
            n (Optional[int], optional): Number of samples. Defaults to None.

        Returns:
            jnp.ndarray: Samples from the target distribution.
        """  # TODO update these docs
        z = self.base_dist._sample(key, condition)
        x = self.bijection.inverse(z, condition)
        return x


class NeuralSplineFlow(Flow):
    def __init__(
        self,
        key: random.PRNGKey,
        base_dist: Distribution,
        cond_dim: int = 0,
        K: int = 10,
        B: int = 5,
        num_layers: int = 5,
        nn_width: int = 40,
        nn_depth: int = 2,
        permute_strategy: Optional[str] = None,
    ):
        """Convenience constructor for Neural spline flow (Durkan et al. 2019;
        https://arxiv.org/abs/1906.04032). Note that the transformation is on
        the interval [-B, B].

        Args:
            key (random.PRNGKey): Random key.
            dim (int): Dimension of the target distribution.
            cond_dim (int, optional): Dimension of extra conditioning variables. Defaults to 0.
            K (int, optional): Number of (inner) spline segments. Defaults to 10.
            B (int, optional): Interval to transform [-B, B]. Defaults to 5.
            num_layers (int, optional): Number of coupling layers. Defaults to 5.
            nn_width (int, optional): Conditioner network width. Defaults to 40.
            nn_depth (int, optional): Conditioner network depth. Defaults to 2.
            permute_strategy (Optional[str], optional): How to permute between layers. Either "flip" or "random". Defaults to "flip" if dim <=2, otherwise "random".
            base_log_prob (Optional[Callable], optional): Log probability in base distribution. Defaults to standard normal.
            base_sample (Optional[Callable], optional): Sample function in base distribution. Defaults to standard normal.
        """  # TODO update docs
        d = base_dist.dim // 2
        if permute_strategy is None:
            permute_strategy = "flip" if base_dist.dim <= 2 else "random"

        permute_key, *layer_keys = random.split(key, num_layers + 1)
        layers = [
            Coupling(
                key=key,
                bijection=RationalQuadraticSpline(K=K, B=B),
                d=d,
                D=base_dist.dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                cond_dim=cond_dim,
            )
            for key in layer_keys
        ]

        layers = intertwine_permute(
            layers, permute_strategy, permute_key, base_dist.dim
        )
        bijection = Chain(layers)
        super().__init__(base_dist, bijection)


class RealNVPFlow(Flow):
    def __init__(
        self,
        key: random.PRNGKey,
        base_dist: Distribution,
        cond_dim: int = 0,
        num_layers: int = 5,
        nn_width: int = 40,
        nn_depth: int = 2,
        permute_strategy: Optional[str] = None,
    ):
        """Convenience constructor for a RealNVP style flow (Dinh et al, 2017;
        https://arxiv.org/abs/1605.08803). Note this implementation differs slightly
        from the original, e.g. it does not use batch normaliziation.

        Args:
            key (random.PRNGKey): Random key.
            dim (int): Dimension of the target distribution.
            cond_dim (int, optional): Dimension of extra conditioning variables. Defaults to 0.
            num_layers (int, optional): Number of coupling layers. Defaults to 5.
            nn_width (int, optional): Conditioner network width. Defaults to 40.
            nn_depth (int, optional): Conditioner network depth. Defaults to 2.
            permute_strategy (Optional[str], optional): How to permute between layers. Either "flip" or "random". Defaults to "flip" if dim <=2, otherwise "random".
            base_log_prob (Optional[Callable], optional): Log probability in base distribution. Defaults to standard normal.
            base_sample (Optional[Callable], optional): Sample function in base distribution. Defaults to standard normal.
        """  # TODO update docs
        d = base_dist.dim // 2
        if permute_strategy is None:
            permute_strategy = "flip" if base_dist.dim <= 2 else "random"

        permute_key, *layer_keys = random.split(key, num_layers + 1)
        layers = [
            Coupling(
                key=key,
                bijection=Affine(),
                d=d,
                D=base_dist.dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                cond_dim=cond_dim,
            )
            for key in layer_keys
        ]

        layers = intertwine_permute(
            layers, permute_strategy, permute_key, base_dist.dim
        )
        bijection = Chain(layers)
        super().__init__(base_dist, bijection)


class BlockNeuralAutoregressiveFlow(Flow):
    def __init__(
        self,
        key: random.PRNGKey,
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
            key (random.PRNGKey): Random key.
            dim (int): Dimension of the target distribution.
            nn_layers (int, optional): Number of layers within autoregressive neural networks. Defaults to 3.
            block_dim (int, optional): Block size in lower triangular blocks of autoregressive neural network. Defaults to 8.
            flow_layers (int, optional): Number of flow layers (1 layer = autoregressive neural network + TanH activation) . Defaults to 1.
            permute_strategy (Optional[str], optional): How to permute between layers. Either "flip" or "random". Defaults to "flip" if dim <=2, otherwise "random".
            base_log_prob (Optional[Callable], optional): Base distribution log probability function. Defaults to standard normal.
            base_sample (Optional[Callable], optional): Base distribution sample function. Defaults to standard normal.
        """
        assert nn_layers >= 2

        if permute_strategy is None:
            permute_strategy = "flip" if base_dist.dim <= 2 else "random"

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
        ]

        bijections = intertwine_permute(
            bijections, permute_strategy, permute_key, base_dist.dim
        )
        bijection = Chain(bijections)
        super().__init__(base_dist, bijection)
