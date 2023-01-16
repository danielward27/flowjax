FAQ
==========

Freezing parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Often it is useful to not train particular parameters. To achieve this we can provide a
``filter_spec`` to :py:func:`~flowjax.train.train_flow`. For example, to avoid
training the base distribution, we could create a ``filter_spec`` as follows

.. code-block:: python

    import equinox as eqx
    import jax.tree_util as jtu
    filter_spec = jtu.tree_map(lambda x: eqx.is_inexact_array(x), flow)
    filter_spec = eqx.tree_at(lambda tree: tree.base_dist, filter_spec, replace=False)

For more information about filtering, see the `equinox documentation <https://docs.kidger.site/equinox/all-of-equinox/>`_.

Standardising variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In general you should consider the form and scales of the target samples. For example, you could define a bijection to carry out the preprocessing, then to transform the flow with the inverse, to "undo" the preprocessing, e.g.

.. code-block:: python

    import jax
    from flowjax.bijections import Affine, Invert
    from flowjax.distributions import Transformed

    preprocess = Affine(-x.mean(axis=0)/x.std(axis=0), 1/x.std(axis=0))
    x_processed = jax.vmap(preprocess.transform)(x)
    flow, losses = train_flow(train_key, flow, x_processed)
    flow = Transformed(flow, Invert(preprocess))  # "undo" the preprocessing

When to JIT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The methods of distributions and bijections are not jitted by default. For example, if you wanted to sample several batches after training, then it is usually worth using jit

.. code-block:: python

    import equinox as eqx
    batch_size = 256
    keys = random.split(random.PRNGKey(0), 5)

    # Often slow - sample not jitted!
    results = []
    for batch_key in keys:
        x = flow.sample(batch_key, n=batch_size)
        results.append(x)

    # Fast - sample jitted!
    results = []
    for batch_key in keys:
        x = eqx.filter_jit(flow.sample)(batch_key, n=batch_size)
        results.append(x)