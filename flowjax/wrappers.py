""":class:`AbstractUnwrappable` objects and utilities.

These are "placeholder" values for specifying custom behaviour for nodes in a pytree.
To apply the custom behaviour, we use :func:`unwrap`, which will replace the
:class:`AbstractUnwrappable` nodes in a pytree with the unwrapped versions.

Examples of custom behaviour includes:
- Reparameterizations: Store and optimize parameters in one (often unbounded) domain
with easy unwrapping to another domain.
- Masking:  Masking out particular values in an array, e.g. to form triangular matrices,
or to enfore particular dependency structures in neural network weights.
- Freezing parameters: Freezing parameters by applying ``jax.lax.stop_gradient`` before
accessing.


If implementing a custom unwrappable, bear in mind:
1) The wrapper should avoid implementing any logic or storing information beyond
    what is required for initialization and unwrapping, as this information will be
    lost when the object is unwrapped.
2) The unwrapping should support broadcasting/vmapped initialization.
"""

from abc import abstractmethod
from typing import Any, Callable, Generic, Iterable, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike

from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.softplus import SoftPlus
from flowjax.utils import _VectorizedBijection, arraylike_to_array

PyTree = Any


T = TypeVar("T")


class AbstractUnwrappable(eqx.Module, Generic[T]):
    """An abstract class representing an unwrappable object.

    Unwrappables generally replace nodes in a pytree, in order to specify some custom
    behaviour to apply upon unwrapping before use. This can be used e.g. to apply
    parameter constraints, such as making scale parameters postive, or applying
    stop_gradient before accessing the parameters.
    """

    def recursive_unwrap(self) -> T:
        """Returns the unwrapped pytree, unwrapping subnodes as required."""
        flat, tree_def = eqx.tree_flatten_one_level(self)
        tree = jax.tree.unflatten(tree_def, unwrap(flat))
        return tree.unwrap()

    @abstractmethod
    def unwrap(self) -> T:
        """Returns the unwrapped pytree, assuming no subnodes need to be unwrapped."""
        pass


class StopGradient(AbstractUnwrappable[T]):
    """Applies stop gradient to all arraylike leaves before unwrapping.

    Useful to mark pytrees (arrays, submodules, etc) as frozen/non-trainable.
    """

    tree: T

    def unwrap(self) -> T:
        differentiable, static = eqx.partition(self.tree, eqx.is_array_like)
        return eqx.combine(lax.stop_gradient(differentiable), static)


def _apply_inverse_and_check_valid(bijection, arr):
    param_inv = _VectorizedBijection(bijection).inverse(arr)
    return eqx.error_if(
        param_inv,
        jnp.logical_and(jnp.isfinite(arr), ~jnp.isfinite(param_inv)),
        "Non-finite value(s) introduced when reparameterizing. This suggests "
        "the parameter vector passed to BijectionReparam was incompatible with "
        f"the bijection used for reparmeterizing ({type(bijection).__name__}).",
    )


class BijectionReparam(AbstractUnwrappable[Array]):
    """Reparameterize a parameter using a bijection.

    When applying unwrap, ``bijection.transform`` is applied. By default, the inverse
    of the bijection is applied when setting the parameter values, and the forward
    transform is applied during unwrapping.

    Args:
        arr: The parameter to reparameterize. If invert_on_init is False, then this can
            be a ``AbstractUnwrappable[Array]``.
        bijection: A bijection whose shape is broadcastable to ``jnp.shape(arr)``.
        invert_on_init: Whether to applying the inverse transformation when
            initializing. Defaults to True.
    """

    arr: Array | AbstractUnwrappable[Array]
    bijection: AbstractBijection

    def __init__(
        self,
        arr: Array | AbstractUnwrappable[Array],
        bijection: AbstractBijection,
        *,
        invert_on_init: bool = True,
    ):
        if invert_on_init:
            self.arr = _apply_inverse_and_check_valid(bijection, arr)
        else:
            if not isinstance(arr, AbstractUnwrappable):
                arr = arraylike_to_array(arr)
            self.arr = arr
        self.bijection = bijection

    def unwrap(self) -> Array:
        return _VectorizedBijection(self.bijection).transform(self.arr)


class Where(AbstractUnwrappable[Array]):
    """Applies jnp.where unpon unwrapping.

    This can be used to construct masks by setting ``cond=mask`` and ``if_false=0``.
    """

    cond: ArrayLike
    if_true: ArrayLike
    if_false: ArrayLike

    def unwrap(self):
        return jnp.where(self.cond, self.if_true, self.if_false)


class WeightNormalization(AbstractUnwrappable[Array]):
    """Applies weight normalization (https://arxiv.org/abs/1602.07868).

    Args:
        weight: The (possibly wrapped) weight matrix.
    """

    weight: Array | AbstractUnwrappable[Array]
    scale: Array = eqx.field(init=False)

    def __init__(self, weight: Array | AbstractUnwrappable[Array]):
        self.weight = weight
        scale_init = 1 / jnp.linalg.norm(unwrap(weight), axis=-1, keepdims=True)
        self.scale = BijectionReparam(scale_init, SoftPlus())

    def unwrap(self) -> Array:
        weight_norms = jnp.linalg.norm(self.weight, axis=-1, keepdims=True)
        return self.scale * self.weight / weight_norms


class Diagonal(AbstractUnwrappable[Array]):
    """Unwraps a vector to a diagonal array."""

    arr: Array | AbstractUnwrappable[Array]

    def unwrap(self):
        return jnp.vectorize(jnp.diag, signature="(a)->(a,a)")(self.arr)


class Lambda(AbstractUnwrappable[T]):
    """Unwrap an object by calling fn with args and kwargs."""

    fn: Callable[..., T]
    args: Iterable
    kwargs: Iterable

    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def unwrap(self) -> T:
        return self.fn(*self.args, **self.kwargs)


def unwrap(tree: PyTree):
    """Unwrap all :class:`AbstractUnwrappable` nodes within a pytree."""
    return jax.tree_util.tree_map(
        f=lambda leaf: (
            leaf.recursive_unwrap() if isinstance(leaf, AbstractUnwrappable) else leaf
        ),
        tree=tree,
        is_leaf=lambda x: isinstance(x, AbstractUnwrappable),
    )
