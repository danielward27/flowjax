""":class:`AbstractUnwrappable` objects and utilities.

These are "placeholder" values for specifying custom behaviour for nodes in a pytree.
Many of these facilitate similar functions to pytorch parameterizations. We use this
for example to apply parameter constraints, masking of parameters etc. To apply the
behaviour, we use :func:`unwrap`, which will replace any :class:`AbstractUnwrappable`
nodes in a pytree with the unwrapped versions.

Unwrapping is automatically called in several places, primarily:

* Prior to calling the bijection methods: ``transform``, ``inverse``,
  ``transform_and_log_det`` and ``inverse_and_log_det``.
* Prior to calling distribution methods: ``log_prob``, ``sample`` and
  ``sample_and_log_prob``.
* Prior to computing the loss functions.

If implementing a custom unwrappable, bear in mind:

* The wrapper should avoid implementing information or logic beyond what is required
  for initialization and unwrapping, as this information will be lost when unwrapping.
* The unwrapping should support broadcasting/vmapped initializations. Otherwise, if
  the unwrappable is created within a batched context, it will fail to unwrap
  correctly.
"""

from abc import abstractmethod
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Int, Scalar

from flowjax.utils import arraylike_to_array

if TYPE_CHECKING:
    from flowjax.bijections import AbstractBijection

import jax
from jax import lax
from jaxtyping import Array, ArrayLike, PyTree

T = TypeVar("T")


def unwrap(tree: PyTree):
    """Unwrap all :class:`AbstractUnwrappable` nodes within a pytree."""
    return jax.tree_util.tree_map(
        f=lambda leaf: (
            leaf.recursive_unwrap() if isinstance(leaf, AbstractUnwrappable) else leaf
        ),
        tree=tree,
        is_leaf=lambda x: isinstance(x, AbstractUnwrappable),
    )


class AbstractUnwrappable(eqx.Module, Generic[T]):
    """An abstract class representing an unwrappable object.

    Unwrappables generally replace nodes in a pytree, in order to specify some custom
    behaviour to apply upon unwrapping before use. This can be used e.g. to apply
    parameter constraints, such as making scale parameters postive, or applying
    stop_gradient before accessing the parameters.

    If ``_dummy`` is set to an array (must have shape ()), this is used for inferring
    vmapped dimensions (and sizes) when calling ``unwrap`` to automatically vecotorize
    the method. In some cases this is important for supporting the case where an
    ``AbstractUnwrappable`` is created within e.g. ``eqx.filter_vmap``.
    """

    _dummy: eqx.AbstractVar[Int[Scalar, ""] | None]

    def recursive_unwrap(self) -> T:
        """Returns the unwrapped pytree, unwrapping subnodes as required."""

        def vectorized_unwrap(unwrappable):
            if unwrappable._dummy is None:
                return unwrappable.unwrap()

            def v_unwrap(unwrappable):
                return unwrappable.unwrap()

            for dim in reversed(unwrappable._dummy.shape):
                v_unwrap = eqx.filter_vmap(v_unwrap, axis_size=dim)
            return v_unwrap(unwrappable)

        flat, tree_def = eqx.tree_flatten_one_level(self)
        tree = jax.tree_util.tree_unflatten(tree_def, unwrap(flat))
        return vectorized_unwrap(tree)

    @abstractmethod
    def unwrap(self) -> T:
        """Returns the unwrapped pytree, assuming no wrapped subnodes exist."""
        pass


class NonTrainable(AbstractUnwrappable[T]):
    """Applies stop gradient to all arraylike leaves before unwrapping.

    See also ``non_trainable``, which is probably a generally prefereable way to achieve
    similar behaviour, which wraps the arraylike leaves directly, rather than the tree.

    Useful to mark pytrees (arrays, submodules, etc) as frozen/non-trainable. We also
    filter out NonTrainable nodes when partitioning parameters for training, or when
    parameterizing bijections in coupling/masked autoregressive flows (transformers).
    """

    tree: T
    _dummy: ClassVar[None] = None

    def unwrap(self) -> T:
        differentiable, static = eqx.partition(self.tree, eqx.is_array_like)
        return eqx.combine(lax.stop_gradient(differentiable), static)


def non_trainable(tree: PyTree):
    """Freezes parameters by wrapping inexact array leaves with ``NonTrainable``.

    Wrapping the arrays rather than the entire tree is often preferable, allowing easier
    access to attributes, compared to wrapping the entire tree.

    Args:
        tree: The pytree.
    """

    def _map_fn(leaf):
        return NonTrainable(leaf) if eqx.is_inexact_array(leaf) else leaf

    return jax.tree_util.tree_map(
        f=_map_fn,
        tree=tree,
        is_leaf=lambda x: isinstance(x, NonTrainable),
    )


def _apply_inverse_and_check_valid(bijection, arr):
    param_inv = bijection._vectorize.inverse(arr)
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
    of the bijection is applied when setting the parameter values.

    Args:
        arr: The parameter to reparameterize. If invert_on_init is False, then this can
            be a ``AbstractUnwrappable[Array]``.
        bijection: A bijection whose shape is broadcastable to ``jnp.shape(arr)``.
        invert_on_init: Whether to apply the inverse transformation when initializing.
            Defaults to True.
    """

    arr: Array | AbstractUnwrappable[Array]
    bijection: "AbstractBijection"
    _dummy: Int[Scalar, ""]

    def __init__(
        self,
        arr: Array | AbstractUnwrappable[Array],
        bijection: "AbstractBijection",
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
        self._dummy = jnp.empty((), int)

    def unwrap(self) -> Array:
        return self.bijection._vectorize.transform(self.arr)


class Where(AbstractUnwrappable[Array]):
    """Applies jnp.where upon unwrapping.

    This can be used to construct masks by setting ``cond=mask`` and ``if_false=0``.
    """

    cond: ArrayLike
    if_true: ArrayLike | AbstractUnwrappable[Array]
    if_false: ArrayLike | AbstractUnwrappable[Array]
    _dummy: ClassVar[None] = None

    def unwrap(self):
        return jnp.where(self.cond, self.if_true, self.if_false)


class WeightNormalization(AbstractUnwrappable[Array]):
    """Applies weight normalization (https://arxiv.org/abs/1602.07868).

    Args:
        weight: The (possibly wrapped) weight matrix.
    """

    weight: Array | AbstractUnwrappable[Array]
    scale: Array | AbstractUnwrappable[Array] = eqx.field(init=False)
    _dummy: ClassVar[None] = None

    def __init__(self, weight: Array | AbstractUnwrappable[Array]):
        from flowjax.bijections import SoftPlus  # Delayed to avoid circular import...

        self.weight = weight
        scale_init = 1 / jnp.linalg.norm(unwrap(weight), axis=-1, keepdims=True)
        self.scale = BijectionReparam(scale_init, SoftPlus())

    def unwrap(self) -> Array:
        weight_norms = jnp.linalg.norm(self.weight, axis=-1, keepdims=True)
        return self.scale * self.weight / weight_norms


class Lambda(AbstractUnwrappable[T]):
    """Unwrap an object by calling fn with (possibly trainable) args and kwargs.

    If the Lambda is created within ``eqx.filter_vmap``, unwrapping is automatically
    vectorized correctly, as long as the vmapped constructor adds leading batch
    dimensions to all arrays in Lambda (the default for ``eqx.filter_vmap``).

    Args:
        fn: Function to call with args, and kwargs.
        *args: Positional arguments to pass to fn.
        **kwargs: Keyword arguments to pass to fn.
    """

    fn: Callable[..., T]
    args: Iterable
    kwargs: dict
    _dummy: Int[Scalar, ""]

    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self._dummy = jnp.empty((), int)

    def unwrap(self) -> T:
        return self.fn(*self.args, **self.kwargs)
