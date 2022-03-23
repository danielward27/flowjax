from flowjax.bijections.abc import ParameterisedBijection
import equinox as eqx
import jax.numpy as jnp
from jax import random
from flowjax.bijections.utils import Chain, intertwine_permute
from flowjax.bijections.abc import Bijection


class Coupling(Bijection, eqx.Module):
    d: int
    D: int
    bijection: ParameterisedBijection
    conditioner: eqx.nn.MLP
    condition_dim: int

    def __init__(
        self,
        key: random.PRNGKey,
        bijection: ParameterisedBijection,
        d: int,
        D: int,
        nn_width: int,
        nn_depth: int,
        condition_dim: int = 0,
    ):
        """Coupling layer implementation.

        Args:
            key (random.PRNGKey): Key
            bijection (ParameterisedBijection): Bijection to be parameterised by the conditioner neural netork.
            d (int): Number of untransformed conditioning variables.
            D (int): Total dimension.
            nn_width (int): Number of nodes in hidden layers.
            nn_depth (int): Number of hidden layers.
        """
        self.d = d
        self.D = D
        self.bijection = bijection
        output_size = self.bijection.num_params(D - d)

        self.conditioner = eqx.nn.MLP(
            in_size=d + condition_dim,
            out_size=output_size,
            width_size=nn_width,
            depth=nn_depth,
            key=key,
        )

        self.condition_dim = condition_dim

    def transform(self, x, condition=jnp.array([])):
        x_cond, x_trans = x[: self.d], x[self.d :]
        cond = jnp.concatenate((x_cond, condition))
        bijection_params = self.conditioner(cond)
        bijection_args = self.bijection.get_args(bijection_params)
        y_trans = self.bijection.transform(x_trans, *bijection_args, condition)
        y = jnp.concatenate((x_cond, y_trans))
        return y

    def transform_and_log_abs_det_jacobian(self, x, condition=jnp.array([])):
        x_cond, x_trans = x[: self.d], x[self.d :]
        cond = jnp.concatenate((x_cond, condition))

        bijection_params = self.conditioner(cond)
        bijection_args = self.bijection.get_args(bijection_params)
        y_trans, log_abs_det = self.bijection.transform_and_log_abs_det_jacobian(
            x_trans, *bijection_args, condition
        )
        y = jnp.concatenate([x_cond, y_trans])
        return y, log_abs_det

    def inverse(self, y: jnp.ndarray, condition=jnp.array([])):
        x_cond, y_trans = y[: self.d], y[self.d :]
        cond = jnp.concatenate((x_cond, condition))
        bijection_params = self.conditioner(cond)
        bijection_args = self.bijection.get_args(bijection_params)
        x_trans = self.bijection.inverse(y_trans, *bijection_args)
        x = jnp.concatenate((x_cond, x_trans))
        return x


class CouplingStack(Chain):
    layers: list
    d: int
    D: int
    condition_dim: int

    def __init__(
        self,
        key: random.PRNGKey,
        bijection: ParameterisedBijection,
        D: int,
        condition_dim: int = 0,
        num_layers: int = 5,
        nn_width: int = 40,
        nn_depth: int = 2,
        permute_strategy: str = "flip",
    ):
        """
        A stack of bijections, alternating a coupling layer, and a permutation.
        To learn a conditional distribution, condition_dim should be >0,
        and conditioning variables should be passed when transforming.
        
        Args:
            key (random.PRNGKey): Random key.
            bijection (ParameterisedBijection): Bijection to transform variables
                in coupling layers.
            D (int): Dimension of the target distribution.
            num_layers (int): Number of layers (1 layer = coupling layer + permute).
            nn_width (int, optional): Conditioner network width. Defaults to 40.
            nn_depth (int, optional): Conditioner network depth. Defaults to 2.
            condition_dim (int, optional): Dimension of additional conditioning
                variables (for learning conditional distributions). Defaults to 0.
        """
        d = D // 2
        key, *subkeys = random.split(key, num_layers + 1)
        layers = [
            Coupling(
                key=subkey,
                bijection=bijection,
                d=d,
                D=D,
                nn_width=nn_width,
                nn_depth=nn_depth,
                condition_dim=condition_dim,
            )
            for subkey in subkeys
        ]

        key, subkey = random.split(key)
        layers = intertwine_permute(layers, permute_strategy, subkey, D)

        self.layers = layers
        self.d = d
        self.D = D
        self.condition_dim = condition_dim
        super().__init__(layers)
