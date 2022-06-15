from typing import Callable, Optional
import jax.numpy as jnp
from numpy import broadcast_shapes
from flowjax.bijections.abc import Bijection
from jax.scipy.stats import norm
from jax import random
import equinox as eqx
import jax
from flowjax.bijections.coupling import Coupling
from flowjax.bijections.rational_quadratic_spline import RationalQuadraticSpline
from flowjax.bijections.affine import Affine
from flowjax.bijections.utils import Chain
from flowjax.bijections.bnaf import BlockAutoregressiveNetwork
from flowjax.bijections.utils import intertwine_permute


class Flow(eqx.Module):
    bijection: Bijection
    target_dim: int
    base_log_prob: Callable
    base_sample: Callable

    def __init__(
        self,
        bijection: Bijection,
        target_dim: int,
        base_log_prob: Optional[Callable] = None,
        base_sample: Optional[Callable] = None,
    ):
        """Form a distribution like object using a base distribution and a
        bijection. Operations are generally assumed to be batched along
        dimension 0.

        Args:
            bijection (Bijection): Bijection mapping from target distribution to
                the base distribution.
            target_dim: (int): Dimension of the target distribution.
            base_log_prob (Optional[Callable], optional): log probability in the base
                distribution. Defaults to standard normal.
            base_sample (Optional[Callable], optional): sample function with signature
                (key : PRNGKey, n : int). Defaults to standard normal.
        """
        self.bijection = bijection
        self.target_dim = target_dim

        if base_log_prob:
            self.base_log_prob = base_log_prob
            self.base_sample = base_sample
        else:
            self.base_log_prob = lambda x: norm.logpdf(x).sum(axis=1)
            self.base_sample = lambda key, n: random.normal(key, (n, target_dim))

    @eqx.filter_jit
    def log_prob(self, x: jnp.ndarray, condition: Optional[jnp.ndarray] = None):
        "Evaluate the log probability of the target distribution."
        x = jnp.atleast_2d(x)

        if condition is not None:
            x, condition = self._broadcast(x, condition)

        z, log_abs_det = jax.vmap(self.bijection.transform_and_log_abs_det_jacobian)(
            x, condition
        )
        p_z = self.base_log_prob(z)
        return p_z + log_abs_det

    @staticmethod
    def _broadcast(x: jnp.ndarray, condition: jnp.ndarray):
        "Broadcast arrays, excluding last axis."
        s = broadcast_shapes(x.shape[:-1], condition.shape[:-1])
        x = jnp.broadcast_to(x, s + (x.shape[-1],))
        condition = jnp.broadcast_to(condition, s + (condition.shape[-1],))
        return x, condition

    @eqx.filter_jit
    def sample(
        self,
        key: random.PRNGKey,
        condition: Optional[jnp.ndarray] = None,
        n: Optional[int] = None,
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
        """

        if condition is not None:
            if condition.ndim == 1:
                assert n is not None, "n must be provided with a vector condition."
                condition = jnp.broadcast_to(condition, (n, condition.shape[0]))
            else:
                assert n is None, "n should not be provided if a matrix of conditioning variables is used."
                n = condition.shape[0]

        z = self.base_sample(key, n)
        x = jax.vmap(self.bijection.inverse)(z, condition)
        return x


class NeuralSplineFlow(Flow):
    def __init__(
        self,
        key: random.PRNGKey,
        target_dim: int,
        condition_dim: int = 0,
        K: int = 10,
        B: int = 5,
        num_layers: int = 5,
        nn_width: int = 40,
        nn_depth: int = 2,
        permute_strategy: Optional[str] = None,
        base_log_prob: Optional[Callable] = None,
        base_sample: Optional[Callable] = None,
    ):
        """Convenience constructor for Neural spline flow (Durkan et al. 2019;
        https://arxiv.org/abs/1906.04032). Note that the transformation is on
        the interval [-B, B].

        Args:
            key (random.PRNGKey): Random key.
            target_dim (int): Dimension of the target distribution.
            condition_dim (int, optional): Dimension of extra conditioning variables. Defaults to 0.
            K (int, optional): Number of (inner) spline segments. Defaults to 10.
            B (int, optional): Interval to transform [-B, B]. Defaults to 5.
            num_layers (int, optional): Number of coupling layers. Defaults to 5.
            nn_width (int, optional): Conditioner network width. Defaults to 40.
            nn_depth (int, optional): Conditioner network depth. Defaults to 2.
            permute_strategy (Optional[str], optional): How to permute between layers. Either "flip" or "random". Defaults to "flip" if target_dim <=2, otherwise "random".
            base_log_prob (Optional[Callable], optional): Log probability in base distribution. Defaults to standard normal.
            base_sample (Optional[Callable], optional): Sample function in base distribution. Defaults to standard normal.
        """
        d = target_dim // 2
        if permute_strategy is None:
            permute_strategy = "flip" if target_dim <= 2 else "random"

        permute_key, *layer_keys = random.split(key, num_layers + 1)
        layers = [
            Coupling(
                key=key,
                bijection=RationalQuadraticSpline(K=K, B=B),
                d=d,
                D=target_dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                condition_dim=condition_dim,
            )
            for key in layer_keys
        ]

        layers = intertwine_permute(layers, permute_strategy, permute_key, target_dim)
        bijection = Chain(layers)

        super().__init__(bijection, target_dim, base_log_prob, base_sample)


class RealNVPFlow(Flow):
    def __init__(
        self,
        key: random.PRNGKey,
        target_dim: int,
        condition_dim: int = 0,
        num_layers: int = 5,
        nn_width: int = 40,
        nn_depth: int = 2,
        permute_strategy: Optional[str] = None,
        base_log_prob: Optional[Callable] = None,
        base_sample: Optional[Callable] = None,
    ):
        """Convenience constructor for a RealNVP style flow (Dinh et al, 2017;
        https://arxiv.org/abs/1605.08803). Note this implementation differs slightly
        from the original, e.g. it does not use batch normaliziation.

        Args:
            key (random.PRNGKey): Random key.
            target_dim (int): Dimension of the target distribution.
            condition_dim (int, optional): Dimension of extra conditioning variables. Defaults to 0.
            num_layers (int, optional): Number of coupling layers. Defaults to 5.
            nn_width (int, optional): Conditioner network width. Defaults to 40.
            nn_depth (int, optional): Conditioner network depth. Defaults to 2.
            permute_strategy (Optional[str], optional): How to permute between layers. Either "flip" or "random". Defaults to "flip" if target_dim <=2, otherwise "random".
            base_log_prob (Optional[Callable], optional): Log probability in base distribution. Defaults to standard normal.
            base_sample (Optional[Callable], optional): Sample function in base distribution. Defaults to standard normal.
        """
        d = target_dim // 2
        if permute_strategy is None:
            permute_strategy = "flip" if target_dim <= 2 else "random"

        permute_key, *layer_keys = random.split(key, num_layers + 1)
        layers = [
            Coupling(
                key=key,
                bijection=Affine(),
                d=d,
                D=target_dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                condition_dim=condition_dim,
            )
            for key in layer_keys
        ]

        layers = intertwine_permute(layers, permute_strategy, permute_key, target_dim)
        bijection = Chain(layers)
        super().__init__(bijection, target_dim, base_log_prob, base_sample)


class BlockNeuralAutoregressiveFlow(Flow):
    def __init__(
        self,
        key: random.PRNGKey,
        target_dim: int,
        condition_dim: int = 0,
        nn_layers: int = 3,
        block_size: tuple = (8, 8),
        flow_layers: int = 1,
        permute_strategy: Optional[str] = None,
        base_log_prob: Optional[Callable] = None,
        base_sample: Optional[Callable] = None,
    ):
        """Convenience constructor for a block neural autoregressive flow
        (https://arxiv.org/abs/1904.04676).

        Args:
            key (random.PRNGKey): Random key.
            target_dim (int): Dimension of the target distribution.
            nn_layers (int, optional): Number of layers within autoregressive neural networks. Defaults to 3.
            block_size (tuple, optional): Block size in lower triangular blocks of autoregressive neural network. Defaults to (8, 8).
            flow_layers (int, optional): Number of flow layers (1 layer = autoregressive neural network + TanH activation) . Defaults to 1.
            permute_strategy (Optional[str], optional): How to permute between layers. Either "flip" or "random". Defaults to "flip" if target_dim <=2, otherwise "random".
            base_log_prob (Optional[Callable], optional): Base distribution log probability function. Defaults to standard normal.
            base_sample (Optional[Callable], optional): Base distribution sample function. Defaults to standard normal.
        """
        assert nn_layers >= 2

        if permute_strategy is None:
            permute_strategy = "flip" if target_dim <= 2 else "random"

        permute_key, *layer_keys = random.split(key, flow_layers + 1)

        bijections = [
            BlockAutoregressiveNetwork(
                key, dim=target_dim, condition_dim=condition_dim,
                n_layers=nn_layers, block_size=block_size
            )
            for key in layer_keys
        ]

        bijections = intertwine_permute(
            bijections, permute_strategy, permute_key, target_dim
        )
        bijection = Chain(bijections)
        super().__init__(bijection, target_dim, base_log_prob, base_sample)
