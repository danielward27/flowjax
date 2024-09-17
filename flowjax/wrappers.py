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


.. note::
    
    If creating a custom unwrappable, remember that unwrapping will generally occur
    after initialization of the model. Because of this, we recommend ensuring that
    the ``unwrap`` method supports unwrapping if the model is constructed in a
    vectorized context, such as ``eqx.filter_vmap``, e.g. through broadcasting or
    vectorization.
"""

from abc import abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.nn import softplus
from jaxtyping import Array, Int, PyTree, Scalar

from flowjax.utils import inv_softplus

T = TypeVar("T")


def unwrap(tree: PyTree):
    """Recursively unwraps all :class:`AbstractUnwrappable` nodes within a pytree.

    This leaves all other nodes unchanged. If nested, the innermost
    ``AbstractUnwrappable`` nodes are unwrapped first.

    Example:
        Enforcing positivity.

        .. doctest::

            >>> from flowjax.wrappers import Parameterize, unwrap
            >>> import jax.numpy as jnp
            >>> params = Parameterize(jnp.exp, jnp.zeros(3))
            >>> unwrap(("abc", 1, params))
            ('abc', 1, Array([1., 1., 1.], dtype=float32))
    """

    def _map_fn(leaf):
        if isinstance(leaf, AbstractUnwrappable):
            # Flatten to ignore until all contained AbstractUnwrappables are unwrapped
            flat, tree_def = eqx.tree_flatten_one_level(leaf)
            tree = jax.tree_util.tree_unflatten(tree_def, unwrap(flat))
            return tree.unwrap()
        return leaf

    return jax.tree_util.tree_map(
        f=_map_fn,
        tree=tree,
        is_leaf=lambda x: isinstance(x, AbstractUnwrappable),
    )


class AbstractUnwrappable(eqx.Module, Generic[T]):
    """An abstract class representing an unwrappable object.

    Unwrappables generally replace nodes in a pytree, in order to specify some custom
    behaviour to apply upon unwrapping before use. This can be used e.g. to apply
    parameter constraints, such as making scale parameters postive, or applying
    stop_gradient before accessing the parameters.
    """

    @abstractmethod
    def unwrap(self) -> T:
        """Returns the unwrapped pytree, assuming no wrapped subnodes exist."""
        pass


class Parameterize(AbstractUnwrappable[T]):
    """Unwrap an object by calling fn with args and kwargs.

    All of fn, args and kwargs may contain trainable parameters. If the Parameterize is
    created within ``eqx.filter_vmap``, unwrapping is automatically vectorized
    correctly, as long as the vmapped constructor adds leading batch
    dimensions to all arrays (the default for ``eqx.filter_vmap``).

    Example:
        .. doctest::

            >>> from flowjax.wrappers import Parameterize, unwrap
            >>> import jax.numpy as jnp
            >>> positive = Parameterize(jnp.exp, jnp.zeros(3))
            >>> unwrap(positive)  # Aplies exp on unwrapping
            Array([1., 1., 1.], dtype=float32)

    Args:
        fn: Callable to call with args, and kwargs.
        *args: Positional arguments to pass to fn.
        **kwargs: Keyword arguments to pass to fn.
    """

    fn: Callable[..., T]
    args: Iterable
    kwargs: dict[str, Any]
    _dummy: Int[Scalar, ""]  # Used to track vectorized construction.

    def __init__(self, fn: Callable, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self._dummy = jnp.empty((), int)

    def unwrap(self) -> T:

        def _unwrap_fn(self):
            return self.fn(*self.args, **self.kwargs)

        for dim in reversed(self._dummy.shape):  # vectorize if constructed under vmap
            _unwrap_fn = eqx.filter_vmap(_unwrap_fn, axis_size=dim)
        return _unwrap_fn(self)


class NonTrainable(AbstractUnwrappable[T]):
    """Applies stop gradient to all arraylike leaves before unwrapping.

    See also :func:`non_trainable`, which is probably a generally prefereable way to
    achieve similar behaviour, which wraps the arraylike leaves directly, rather than
    the tree. Useful to mark pytrees (arrays, submodules, etc) as frozen/non-trainable.
    We also filter out NonTrainable nodes when partitioning parameters for training,
    or when parameterizing bijections in coupling/masked autoregressive flows
    (transformers).
    """

    tree: T

    def unwrap(self) -> T:
        differentiable, static = eqx.partition(self.tree, eqx.is_array_like)
        return eqx.combine(lax.stop_gradient(differentiable), static)


def non_trainable(tree: PyTree):
    """Freezes parameters by wrapping inexact array leaves with :class:`NonTrainable`.

    .. note::

        Regularization is likely to apply before unwrapping. To avoid regularization
        impacting non-trainable parameters, they should be filtered out,
        for example using:

        .. code-block:: python

            >>> eqx.partition(
            ...     ...,
            ...     is_leaf=lambda leaf: isinstance(leaf, wrappers.NonTrainable),
            ... )


        This is done in both :func:`~flowjax.train.fit_to_data` and
        :func:`~flowjax.train.fit_to_variational_target`.

    Wrapping the arrays rather than the entire tree is often preferable, allowing easier
    access to attributes compared to wrapping the entire tree.

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


class WeightNormalization(AbstractUnwrappable[Array]):
    """Applies weight normalization (https://arxiv.org/abs/1602.07868).

    Args:
        weight: The (possibly wrapped) weight matrix.
    """

    weight: Array | AbstractUnwrappable[Array]
    scale: Array | AbstractUnwrappable[Array] = eqx.field(init=False)

    def __init__(self, weight: Array | AbstractUnwrappable[Array]):
        self.weight = weight
        scale_init = 1 / jnp.linalg.norm(unwrap(weight), axis=-1, keepdims=True)
        self.scale = Parameterize(softplus, inv_softplus(scale_init))

    def unwrap(self) -> Array:
        weight_norms = jnp.linalg.norm(self.weight, axis=-1, keepdims=True)
        return self.scale * self.weight / weight_norms
