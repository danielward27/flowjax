FAQ
==========

Freezing parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Often it is useful to not train particular parameters. The easiest way to achieve this
is to use :func:`flowjax.wrappers.non_trainable`. This will wrap the inexact array
leaves with :class:`flowjax.wrappers.NonTrainable`, which will apply ``stop_gradient``
when unwrapping the parameters. For commonly used distribution and bijection methods,
unwrapping is applied automatically. For example

.. doctest::
    
    >>> from flowjax.distributions import Normal
    >>> from flowjax.wrappers import non_trainable
    >>> dist = Normal()
    >>> dist = non_trainable(dist)

To mark part of a tree as frozen, use ``non_trainable`` with e.g. 
``equinox.tree_at`` or ``jax.tree_map``.


Standardizing variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In general you should consider the form and scales of the target samples. For example, you could define a bijection to carry out the preprocessing, then to transform the flow with the inverse, to "undo" the preprocessing, e.g.

.. testsetup::

    from flowjax.distributions import Normal
    from flowjax.train import fit_to_data
    import jax.numpy as jnp
    import jax.random as jr
    
    key = jr.key(0)
    x = jr.normal(key, (1000,3))
    flow = Normal(jnp.ones(3))

.. doctest::

    >>> import jax
    >>> from flowjax.bijections import Affine, Invert
    >>> from flowjax.distributions import Transformed
    >>> preprocess = Affine(-x.mean(axis=0)/x.std(axis=0), 1/x.std(axis=0))
    >>> x_processed = jax.vmap(preprocess.transform)(x)
    >>> flow, losses = fit_to_data(key, dist=flow, x=x_processed) # doctest: +SKIP
    >>> flow = Transformed(flow, Invert(preprocess))  # "undo" the preprocessing
    

When to JIT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The methods of distributions and bijections are not jitted by default. For example, if you wanted to sample several batches after training, then it is usually worth using jit

.. testsetup::

    from flowjax.distributions import Normal
    import jax.numpy as jnp
    import jax.random as jr
    
    key = jr.key(0)
    x = jr.normal(key, (256,3))
    flow = Normal(jnp.ones(3))

.. doctest::

    >>> import equinox as eqx
    >>> import jax.random as jr

    >>> batch_size = 256
    >>> keys = jr.split(jr.key(0), 5)

    >>> # Often slow - sample not jitted!
    >>> results = []
    >>> for batch_key in keys:
    ...     x = flow.sample(batch_key, (batch_size,))
    ...     results.append(x)

    >>> # Fast - sample jitted!
    >>> results = []
    >>> for batch_key in keys:
    ...     x = eqx.filter_jit(flow.sample)(batch_key, (batch_size,))
    ...     results.append(x)
    

Serialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As the distributions and bijections are equinox modules, we can serialize/deserialize
them using the same method outlined in the
`equinox documentation <https://docs.kidger.site/equinox/api/serialisation/>`_.


Runtime type checking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you want to enable runtime type checking we can use
`jaxtyping <https://github.com/patrick-kidger/jaxtyping>`_ and a typechecker such as
`beartype <https://github.com/beartype/beartype>`_. Below is an example using
jaxtypings import hook

.. doctest::
    
    >>> from jaxtyping import install_import_hook

    >>> with install_import_hook("flowjax", "beartype.beartype"):
    ...    from flowjax import bijections as bij

    >>> exp = bij.Exp(shape=2)  # Raises a helpful error as 2 is not a tuple
    