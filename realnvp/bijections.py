from abc import ABC, abstractmethod
import jax.numpy as jnp
import equinox as eqx
from jax import random
import jax

class Bijection(ABC):

    @abstractmethod
    def transform(self, x, *args):
        pass

    @abstractmethod
    def transform_and_log_abs_det_jacobian(self, x, *args):
        pass

    @abstractmethod
    def inverse(self, x, *args):
        pass

    @abstractmethod
    def num_params(self, dim : int):
        "Number of parameters required for bijection, with dimension d"
        pass
    

class Affine(Bijection):
    def transform(self, x, params):
        loc, log_scale = params.split(2)
        return x*jnp.exp(log_scale) + loc
        
    def transform_and_log_abs_det_jacobian(self, x, params):
        loc, log_scale = params.split(2)
        return x*jnp.exp(log_scale) + loc, jnp.sum(log_scale)

    def inverse(self, y, params):
        loc, log_scale = params.split(2)
        return (y-loc)/jnp.exp(log_scale)

    def num_params(self, dim : int):
        return dim*2


class Permute(Bijection):
    permutation: jnp.ndarray  # with indices 0-d
    inverse_permutation: jnp.ndarray

    def __init__(self, permutation):
        assert (permutation.sort() == jnp.arange(len(permutation))).all()
        self.permutation = permutation 
        self.inverse_permutation = jnp.argsort(permutation)  

    def transform(self, x):
        return x[self.permutation]

    def transform_and_log_abs_det_jacobian(self, x):
        return x[self.permutation], jnp.array([0])

    def inverse(self, y):
        return y[self.inverse_permutation]

    def num_params(self, dim : int):
        return 0


class CouplingLayer(eqx.Module):
    d : int  # Where to partition
    D : int # Total dimension
    bijection : Bijection
    conditioner : eqx.nn.MLP

    def __init__(
        self,
        key : random.PRNGKey,
        d : int,
        D : int,
        conditioner_width : int,
        conditioner_depth : int,
        bijection = None):
        self.d = d
        self.D = D
        self.bijection = bijection if bijection else Affine()
        output_size = self.bijection.num_params(D-d)
        self.conditioner =  eqx.nn.MLP(
            d, output_size, conditioner_width, conditioner_depth, key=key)

    def __call__(self, x: jnp.ndarray):
        return self.transform_and_log_abs_det_jacobian(x)    
    
    def transform_and_log_abs_det_jacobian(self, x):
        x_cond, x_trans = x[:self.d], x[self.d:]
        bijection_params = self.conditioner(x_cond)
        y_trans, log_abs_det = self.bijection.transform_and_log_abs_det_jacobian(
            x_trans, bijection_params)
        y = jnp.concatenate([x_cond, y_trans])
        return y, log_abs_det

    def transform(self, x):
        x_cond, x_trans = x[:self.d], x[self.d:]
        bijection_params = self.conditioner(x_cond)
        y_trans = self.bijection.transform(x_trans, bijection_params)
        y = jnp.concatenate([x_cond, y_trans])
        return y

    def inverse(self, y: jnp.ndarray):
        x_cond, y_trans = y[:self.d], y[self.d:]
        bijection_params = self.conditioner(x_cond)
        x_trans = self.bijection.inverse(y_trans, bijection_params)
        x = jnp.concatenate([x_cond, x_trans])
        return x


class RealNVP(eqx.Module):
    layers : list

    def __init__(
        self,
        key : random.PRNGKey,
        D : int,
        conditioner_width : int,
        conditioner_depth : int,
        num_layers : int,  # add option for other bijections?
        ):
        
        layers = []
        ds = [round(jnp.floor(D/2).item()), round(jnp.ceil(D/2).item())]
        permutation = jnp.flip(jnp.arange(D))
        for i in range(num_layers):
            key, subkey = random.split(key)
            d = ds[0] if i % 2 == 0 else ds[1]
            layers.extend(
                [CouplingLayer(
                    key, d, D, conditioner_width, conditioner_depth, Affine()),
                Permute(permutation)]
            )
        self.layers = layers

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

