from flowjax.bijections.abc import Bijection
import jax.numpy as jnp
import equinox as eqx
from jax import random
from jax.random import KeyArray
from typing import Optional, List
from flowjax.bijections.abc import Bijection
from flowjax.utils import Array


class Chain(Bijection):
    bijections: List[Bijection]
    cond_dim: int

    def __init__(self, bijections: List[Bijection]):
        """Chain together bijections to form another bijection.

        Args:
            bijections (List[Bijection]): List of bijections.
        """
        self.bijections = bijections
        self.cond_dim = max([b.cond_dim for b in bijections])

    def transform(self, x, condition=None):
        z = x
        for bijection in self.bijections:
            z = bijection.transform(z, condition)
        return z

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        log_abs_det_jac = 0
        z = x
        for bijection in self.bijections:
            z, log_abs_det_jac_i = bijection.transform_and_log_abs_det_jacobian(
                z, condition
            )
            log_abs_det_jac += log_abs_det_jac_i
        return z, log_abs_det_jac

    def inverse(self, z: Array, condition=None):
        x = z
        for bijection in reversed(self.bijections):
            x = bijection.inverse(x, condition)
        return x


class Permute(Bijection):
    permutation: Array
    inverse_permutation: Array
    cond_dim: int

    def __init__(self, permutation: Array):
        """Permutation transformation. Note condition is ignored.

        Args:
            permutation (Array): Indexes 0-(dim-1) representing new order.
        """
        assert (permutation.sort() == jnp.arange(len(permutation))).all()
        self.permutation = permutation
        self.inverse_permutation = jnp.argsort(permutation)
        self.cond_dim = 0

    def transform(self, x, condition=None):
        return x[self.permutation]

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        return x[self.permutation], jnp.array(0)

    def inverse(self, y, condition=None):
        return y[self.inverse_permutation]


class Flip(Bijection):
    """Flip the input array. Condition argument is ignored."""

    cond_dim: int = 0

    def transform(self, x, condition=None):
        return jnp.flip(x)

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        return jnp.flip(x), jnp.array(0)

    def inverse(self, y, condition=None):
        return jnp.flip(y)


def intertwine_flip(bijections: List[Bijection]) -> List[Bijection]:
    """Given a list of bijections, add 'flips' between layers. i.e.
    with bijections [a,b,c], returns [a, flip, b, flip, c]."""
    new_bijections = []
    for b in bijections[:-1]:
        new_bijections.extend([b, Flip()])
    new_bijections.append(bijections[-1])
    return bijections


def intertwine_random_permutation(
    key: KeyArray, bijections: List[Bijection], dim: int
) -> List[Bijection]:
    """Given a list of bijections, add permutations between layers. i.e.
    with bijections [a,b,c], returns [a, perm1, b, perm2, c].

    Args:
        key (KeyArray, optional): Random key (ignored for flip). Defaults to None.
        bijection_list (List[Bijection]): List of bijections.
        dim (int, optional): Dimension (ignored for flip). Defaults to None.
    """
    permutations = jnp.row_stack([jnp.arange(dim) for _ in range(len(bijections) - 1)])
    permutations = random.permutation(key, permutations, 1, True)

    new_bijections = []
    for bijection, permutation in zip(bijections[:-1], permutations):
        new_bijections.extend([bijection, Permute(permutation)])
    new_bijections.append(bijections[-1])
    return new_bijections


def intertwine_permute(
    key: KeyArray,
    bijections: List[Bijection],
    dim: int,
    permute_strategy: Optional[str] = None,
):
    if dim > 1:
        if permute_strategy is None:
            permute_strategy = "flip" if dim == 2 else "random"
        if permute_strategy == "flip":
            bijections = intertwine_flip(bijections)
        elif permute_strategy == "random":
            bijections = intertwine_random_permutation(key, bijections, dim)
        else:
            raise ValueError("Permute strategy should be 'flip' or 'random' if provided.")
    return bijections
