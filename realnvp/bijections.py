import jax.numpy as jnp
import equinox as eqx
from jax import random
import jax
from scipy.misc import derivative
from realnvp.bijection_abcs import Bijection, ParameterisedBijection




class Affine(ParameterisedBijection):
    def transform(self, x, loc, log_scale):
        return x * jnp.exp(log_scale) + loc

    def transform_and_log_abs_det_jacobian(self, x, loc, log_scale):
        return x * jnp.exp(log_scale) + loc, jnp.sum(log_scale)

    def inverse(self, y, loc, log_scale):
        return (y - loc) / jnp.exp(log_scale)

    def num_params(self, dim: int):
        return dim * 2

    def get_args(self, params):
        loc, log_scale = params.split(2)
        return loc, log_scale


class Permute(Bijection):
    permutation: jnp.ndarray  # with indices 
    inverse_permutation: jnp.ndarray

    def __init__(self, permutation):
        """Permutation transformation.

        Args:
            permutation (jnp.ndarray): Indexes 0-(dim-1) representing new order.
        """
        assert (permutation.sort() == jnp.arange(len(permutation))).all()
        self.permutation = permutation
        self.inverse_permutation = jnp.argsort(permutation)

    def transform(self, x):
        return x[self.permutation]

    def transform_and_log_abs_det_jacobian(self, x):
        return x[self.permutation], jnp.array([0])

    def inverse(self, y):
        return y[self.inverse_permutation]

    def num_params(self, dim: int):
        return 0

    def get_args(self, params):
        pass


class CouplingLayer(eqx.Module):
    d: int  # Where to partition
    D: int  # Total dimension
    bijection: Bijection
    conditioner: eqx.nn.MLP

    def __init__(
        self,
        key: random.PRNGKey,
        d: int,
        D: int,
        conditioner_width: int,
        conditioner_depth: int,
        bijection=None,
    ):
        self.d = d
        self.D = D
        self.bijection = bijection if bijection else Affine()
        output_size = self.bijection.num_params(D - d)
        self.conditioner = eqx.nn.MLP(
            d, output_size, conditioner_width, conditioner_depth, key=key
        )

    def __call__(self, x: jnp.ndarray):
        return self.transform_and_log_abs_det_jacobian(x)

    def transform_and_log_abs_det_jacobian(self, x):
        x_cond, x_trans = x[: self.d], x[self.d :]
        bijection_params = self.conditioner(x_cond)
        bijection_args = self.bijection.get_args(bijection_params)
        y_trans, log_abs_det = self.bijection.transform_and_log_abs_det_jacobian(
            x_trans, *bijection_args
        )
        y = jnp.concatenate([x_cond, y_trans])
        return y, log_abs_det

    def transform(self, x):
        x_cond, x_trans = x[: self.d], x[self.d :]
        bijection_params = self.conditioner(x_cond)
        bijection_args = self.bijection.get_args(bijection_params)
        y_trans = self.bijection.transform(x_trans, *bijection_args)
        y = jnp.concatenate([x_cond, y_trans])
        return y

    def inverse(self, y: jnp.ndarray):
        x_cond, y_trans = y[: self.d], y[self.d :]
        bijection_params = self.conditioner(x_cond)
        bijection_args = self.bijection.get_args(bijection_params)
        x_trans = self.bijection.inverse(y_trans, *bijection_args)
        x = jnp.concatenate([x_cond, x_trans])
        return x


class RealNVP(eqx.Module):
    layers: list

    def __init__(
        self,
        key: random.PRNGKey,
        D: int,
        conditioner_width: int,
        conditioner_depth: int,
        num_layers: int,  # add option for other bijections?
    ):

        layers = []
        ds = [round(jnp.floor(D / 2).item()), round(jnp.ceil(D / 2).item())]
        permutation = jnp.flip(jnp.arange(D))
        for i in range(num_layers):
            key, subkey = random.split(key)
            d = ds[0] if i % 2 == 0 else ds[1]
            layers.extend(
                [
                    CouplingLayer(
                        key, d, D, conditioner_width, conditioner_depth, Affine()
                    ),
                    Permute(permutation)
                ]
            )
        self.layers = layers[:-2]  # remove last leakyRelu and permute

    def transform(self, x):
        z = x
        for layer in self.layers:
            z = layer.transform(z)
        return z

    def transform_and_log_abs_det_jacobian(self, x):
        log_abs_det_jac = 0
        z = x
        for layer in self.layers:
            z, log_abs_det_jac_i = layer.transform_and_log_abs_det_jacobian(z)
            log_abs_det_jac += log_abs_det_jac_i
        return z, log_abs_det_jac

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x):
        """Log probability in target distribution, assuming a standard normal
        base distribution."""
        z, log_abs_det = self.transform_and_log_abs_det_jacobian(x)
        p_z = jax.scipy.stats.norm.logpdf(z)
        return (p_z + log_abs_det).mean()



class RationalQuadraticSpline(ParameterisedBijection):
    K : int
    B : float
    _pos_pad : jnp.ndarray  # End knots and beyond

    def __init__(self, K, B):
        self.K = K
        self.B = B
        pos_pad = jnp.zeros(self.K+4)
        pad_idxs = jnp.array([0, 1, -2, -1])
        pad_vals = jnp.array([-B*1e4, -B, B, B*1e4]) # Avoids jax control flow for identity tails
        pos_pad = pos_pad.at[pad_idxs].set(pad_vals)
        self._pos_pad = pos_pad


    @property  # TODO update to Buffer when introduced to eqx: https://github.com/patrick-kidger/equinox/issues/31
    def pos_pad(self):
        return jax.lax.stop_gradient(self._pos_pad)

    def transform(self, x, x_pos, y_pos, derivatives):
        # Roughly following notation from Neural Spline Flow paper (univariate)
        k = jnp.where((x <= x_pos[1:]) & (x > x_pos[:-1]), size=1)[0]
        yk, yk1, xk, xk1 = y_pos[k], y_pos[k+1], x_pos[k], x_pos[k+1]
        sk = (yk1 - yk) / (xk1 - xk)
        xi = (x - xk) / (xk1 - xk)
        num = (yk1 - yk)*(sk*xi**2 + derivatives[k]*xi*(1-xi))
        den = sk + (derivatives[k+1] + derivatives[k] - 2*sk)*xi*(1-xi)
        return yk + num/den

    def inverse(self, y, x_pos, y_pos, derivatives):
        k = jnp.where((y <= y_pos[1:]) & (y > y_pos[:-1]), size=1)[0]
        xk, yk = x_pos[k], y_pos[k]
        sk = (y_pos[k+1] - yk) / (x_pos[k+1] - xk)
        y_delta_s_term = (y-yk)*(derivatives[k+1] + derivatives[k] - 2*sk)
        a = (y_pos[k+1] - yk)*(sk - derivatives[k]) + y_delta_s_term
        b = (y_pos[k+1] - yk)*derivatives[k] - y_delta_s_term
        c = -sk*(y - yk)
        xi = (2*c)/(-b - jnp.sqrt(b**2 - 4*a*c))  # TODO how to deterimine correct root?
        x = xi*(x_pos[k+1] - xk) + xk
        return x

    def transform_and_log_abs_det_jacobian(self, x, x_pos, y_pos, derivatives):
        raise NotImplementedError

    def num_params(self, dim: int = None):
        return self.K*3-1

    def get_args(self, params):
        widths = jax.nn.softmax(params[:self.K])*2*self.B
        heights = jax.nn.softmax(params[self.K:self.K*2])*2*self.B
        derivatives = jax.nn.softmax(params[self.K*2:])
        derivatives = jnp.pad(derivatives, 2, constant_values=1)
        x_pos = jnp.cumsum(widths) - self.B
        x_pos = self.pos_pad.at[2:-2].set(x_pos)
        y_pos = jnp.cumsum(heights) - self.B
        y_pos = self.pos_pad.at[2:-2].set(y_pos)
        return x_pos, y_pos, derivatives
    

# %%
import jax.numpy as jnp
z = jnp.zeros(3)
jnp.pad(z, 2, mode="linear_ramp", end_values=(-10, 10))

# %%
jnp.pad(z, 2, constant_values=(-10, -5))

# %%
