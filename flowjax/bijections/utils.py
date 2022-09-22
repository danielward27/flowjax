from flowjax.bijections.abc import Bijection
import jax.numpy as jnp
from jax import random
from jax.random import KeyArray
from flowjax.utils import Array
from typing import List, Sequence, Tuple, Union


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

    def transform(self, x, condition = None):
        return self.bijection.inverse(x, condition)

    def transform_and_log_abs_det_jacobian(self, x, condition = None):
        return self.bijection.inverse_and_log_abs_det_jacobian(x, condition)

    def inverse(self, y, condition = None):
        return self.bijection.transform(y, condition)

    def inverse_and_log_abs_det_jacobian(self, y, condition = None):
        return self.bijection.transform_and_log_abs_det_jacobian(y, condition)


class Chain(Bijection):
    bijections: Tuple[Bijection]
    cond_dim: int

    def __init__(self, bijections: Sequence[Bijection]):
        """Chain together bijections to form another bijection.

        Args:
            bijections (Sequence[Bijection]): Sequence of bijections.
        """
        self.bijections = tuple(bijections)
        self.cond_dim = max([b.cond_dim for b in bijections])

    def transform(self, x, condition=None):
        for bijection in self.bijections:
            x = bijection.transform(x, condition)
        return x

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        log_abs_det_jac = 0
        for bijection in self.bijections:
            x, log_abs_det_jac_i = bijection.transform_and_log_abs_det_jacobian(
                x, condition
            )
            log_abs_det_jac += log_abs_det_jac_i
        return x, log_abs_det_jac

    def inverse(self, y: Array, condition=None):
        for bijection in reversed(self.bijections):
            y = bijection.inverse(y, condition)
        return y

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        log_abs_det_jac = 0
        for bijection in reversed(self.bijections):
            y, log_abs_det_jac_i = bijection.inverse_and_log_abs_det_jacobian(
                y, condition
            )
            log_abs_det_jac += log_abs_det_jac_i
        return y, log_abs_det_jac

    def __getitem__(self, i: Union[int, slice]) -> Bijection:
        if isinstance(i, int):
            return self.bijections[i]
        elif isinstance(i, slice):
            return Chain(self.bijections[i])
        else:
            raise TypeError(f"Indexing with type {type(i)} is not supported.")

    def __iter__(self):
        yield from self.bijections

    def __len__(self):
        return len(self.bijections)


class Permute(Bijection):
    permutation: Array
    inverse_permutation: Array
    cond_dim: int

    def __init__(self, permutation: Array):
        """Permutation transformation. condition is ignored.

        Args:
            permutation (Array): Indexes 0-(dim-1) representing new order.
        """
        if not (permutation.sort() == jnp.arange(len(permutation))).all():
            raise ValueError("Invalid permutation array provided.")
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


def intertwine_flip(bijections: Sequence[Bijection]) -> List[Bijection]:
    """Given a sequence of bijections, add 'flips' between layers, i.e.
    with bijections [a,b,c], returns [a, flip, b, flip, c].

    Args:
        bijections (Sequence[Bijection]): Sequence of bijections.

    Returns:
        List[Bijection]: List of bijections with flips inbetween.
    """
    new_bijections = []
    for b in bijections[:-1]:
        new_bijections.extend([b, Flip()])
    new_bijections.append(bijections[-1])
    return new_bijections


def intertwine_random_permutation(
    key: KeyArray, bijections: Sequence[Bijection], dim: int
) -> List[Bijection]:
    """Given a list of bijections, add random permutations between layers. i.e.
    with bijections [a,b,c], returns [a, perm1, b, perm2, c].

    Args:
        key (KeyArray): Jax PRNGKey
        bijections (Sequence[Bijection]): Sequence of bijections.
        dim (int): Dimension.

    Returns:
        List[Bijection]: List of bijections with random permutations inbetween.
    """
    new_bijections = []
    for bijection in bijections[:-1]:
        key, subkey = random.split(key)
        perm = random.permutation(subkey, jnp.arange(dim))
        new_bijections.extend([bijection, Permute(perm)])
        
    new_bijections.append(bijections[-1])
    return new_bijections
