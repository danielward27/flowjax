from typing import Union

import equinox as eqx
import jax.numpy as jnp
from jax.experimental import checkify

from flowjax.bijections import Bijection
from flowjax.transformers import Transformer
from flowjax.utils import Array


class Invert(Bijection):
    bijection: Bijection
    cond_dim: int

    def __init__(self, bijection: Bijection):
        """Invert a bijection, such that the transform methods become the inverse methods and vice versa.
        Note that in general, we define bijections such that the forward methods are preffered, i.e.
        faster/actually implemented. For training flows, we generally want the inverse method (used in
        density evaluation), to be faster. Hence it is often useful to use this class to achieve this aim.

        Args:
            bijection (Bijection): Bijection to "invert".
        """
        self.bijection = bijection
        self.cond_dim = bijection.cond_dim

    def transform(self, x, condition=None):
        return self.bijection.inverse(x, condition)

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        return self.bijection.inverse_and_log_abs_det_jacobian(x, condition)

    def inverse(self, y, condition=None):
        return self.bijection.transform(y, condition)

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        return self.bijection.transform_and_log_abs_det_jacobian(y, condition)


class Permute(Bijection):
    permutation: Array
    inverse_permutation: Array
    cond_dim: int

    def __init__(self, permutation: Array):
        """Permutation transformation. condition is ignored.

        Args:
            permutation (Array): Indexes 0-(dim-1) representing new order.
        """
        checkify.check(
            (permutation.sort() == jnp.arange(len(permutation))).all(),
            "Invalid permutation array provided.",
        )
        self.permutation = permutation
        self.inverse_permutation = jnp.argsort(permutation)
        self.cond_dim = 0

    def transform(self, x, condition=None):
        return x[self.permutation]

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        return x[self.permutation], jnp.array(0)

    def inverse(self, y, condition=None):
        return y[self.inverse_permutation]

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        return y[self.inverse_permutation], jnp.array(0)


class Flip(Bijection):
    """Flip the input array. Condition argument is ignored."""

    cond_dim: int = 0

    def transform(self, x, condition=None):
        return jnp.flip(x)

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        return jnp.flip(x), jnp.array(0)

    def inverse(self, y, condition=None):
        return jnp.flip(y)

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        return jnp.flip(y), jnp.array(0)


class TransformerToBijection(Bijection):
    cond_dim: int = 0
    params: Array
    transformer: Transformer

    def __init__(self, transformer: Transformer, *, params: Array):
        """Convert Transformer object to Bijection object.

        Args:
            transformer (Transformer): Transformer.
            params (Array): Unconstrained parameter vector from which the args can be constructed, using `transformer.get_args(params)`.
        """
        # Note, params is key word only, as perhaps we will want to support
        # construction from args too, although this is not implemented yet.
        self.params = params
        self.transformer = transformer

    def transform(self, x: Array, condition=None):
        args = self.transformer.get_args(self.params)
        return self.transformer.transform(x, *args)

    def transform_and_log_abs_det_jacobian(self, x: Array, condition=None):
        args = self.transformer.get_args(self.params)
        return self.transformer.transform_and_log_abs_det_jacobian(x, *args)

    def inverse(self, y: Array, condition=None):
        args = self.transformer.get_args(self.params)
        return self.transformer.inverse(y, *args)

    def inverse_and_log_abs_det_jacobian(self, y: Array, condition=None):
        args = self.transformer.get_args(self.params)
        return self.transformer.inverse_and_log_abs_det_jacobian(y, *args)


class Partial(Bijection):
    """Applies bijection to specific indices of an input."""

    cond_dim: int
    bijection: Array
    idxs: Union[int, slice, Array]

    def __init__(self, bijection: Bijection, idxs):
        """
        Args:
            bijection (Bijection): Bijection that is compatible with the subset of x indexed by idxs.
            idxs: Indices (Integer, a slice, or an ndarray with integer/bool dtype) of the transformed portion. If a multidimensional array is provided, the array is flattened.
        """
        self.bijection = bijection
        self.cond_dim = self.bijection.cond_dim

        if not isinstance(idxs, slice):
            idxs = jnp.array(idxs).ravel()

            if jnp.issubdtype(idxs, jnp.integer):
                idxs = jnp.unique(idxs)

        self.idxs = idxs

    def transform(self, x: Array, condition=None):
        y = self.bijection.transform(x[self.idxs], condition)
        return x.at[self.idxs].set(y)

    def transform_and_log_abs_det_jacobian(self, x: Array, condition=None):
        y, log_det = self.bijection.transform_and_log_abs_det_jacobian(
            x[self.idxs], condition
        )
        return x.at[self.idxs].set(y), log_det

    def inverse(self, y: Array, condition=None) -> Array:
        x = self.bijection.inverse(y[self.idxs], condition)
        return y.at[self.idxs].set(x)

    def inverse_and_log_abs_det_jacobian(self, y: Array, condition=None) -> Array:
        x, log_det = self.bijection.inverse_and_log_abs_det_jacobian(
            y[self.idxs], condition
        )
        return y.at[self.idxs].set(x), log_det


class EmbedCondition(Bijection):
    bijection: Bijection
    embedding_net: eqx.Module
    cond_dim: int

    def __init__(
        self, bijection: Bijection, embedding_net: eqx.Module, cond_dim: int
    ) -> None:
        """Use an embedding network to reduce the dimensionality of the conditioning variable.
        The returned bijection has cond_dim equal to the raw condition size.

        Args:
            bijection (Bijection): Bijection with ``bijection.cond_dim`` equal to the embedded size.
            embedding_net (eqx.Module): A callable equinox module that embeds a conditioning variable to size ``bijection.cond_dim``.
            cond_dim (int): The dimension of the raw conditioning variable.
        """
        self.bijection = bijection
        self.embedding_net = embedding_net
        self.cond_dim = cond_dim

    def transform(self, x, condition=None):
        condition = self.embedding_net(condition)
        return self.bijection.transform(x, condition)

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        condition = self.embedding_net(condition)
        return self.bijection.transform_and_log_abs_det_jacobian(x, condition)

    def inverse(self, y, condition=None):
        condition = self.embedding_net(condition)
        return self.bijection.inverse(y, condition)

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        condition = self.embedding_net(condition)
        return self.bijection.inverse_and_log_abs_det_jacobian(y, condition)
