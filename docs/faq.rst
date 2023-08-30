FAQ
==========

Freezing parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Often it is useful to not train particular parameters. To achieve this we can provide a
``filter_spec`` to :py:func:`~flowjax.train.data_fit.fit_to_data`. For example, to avoid
training the base distribution, we could create a ``filter_spec`` as follows

.. testsetup::

    from flowjax.distributions import Normal    
    flow = Normal()

.. doctest::

    >>> import equinox as eqx
    >>> import jax.tree_util as jtu
    >>> filter_spec = jtu.tree_map(lambda x: eqx.is_inexact_array(x), flow)
    >>> filter_spec = eqx.tree_at(lambda tree: tree.base_dist, filter_spec, replace=False)

For more information about filtering, see the `equinox documentation <https://docs.kidger.site/equinox/all-of-equinox/>`_.

Standardising variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In general you should consider the form and scales of the target samples. For example, you could define a bijection to carry out the preprocessing, then to transform the flow with the inverse, to "undo" the preprocessing, e.g.

.. testsetup::

    from flowjax.distributions import Normal
    from flowjax.train import fit_to_data
    import jax.numpy as jnp
    import jax.random as jr
    
    key = jr.PRNGKey(0)
    x = jr.normal(key, (1000,3))
    flow = Normal(jnp.ones(3))

.. doctest::

    >>> import jax
    >>> from flowjax.bijections import Affine, Invert
    >>> from flowjax.distributions import Transformed
    >>> preprocess = Affine(-x.mean(axis=0)/x.std(axis=0), 1/x.std(axis=0))
    >>> x_processed = jax.vmap(preprocess.transform)(x)
    >>> flow, losses = fit_to_data(key, flow, x_processed) # doctest: +SKIP
    >>> flow = Transformed(flow, Invert(preprocess))  # "undo" the preprocessing
    

When to JIT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The methods of distributions and bijections are not jitted by default. For example, if you wanted to sample several batches after training, then it is usually worth using jit

.. testsetup::

    from flowjax.distributions import Normal
    import jax.numpy as jnp
    import jax.random as jr
    
    key = jr.PRNGKey(0)
    x = jr.normal(key, (256,3))
    flow = Normal(jnp.ones(3))

.. doctest::

    >>> import equinox as eqx
    >>> import jax.random as jr

    >>> batch_size = 256
    >>> keys = jr.split(jr.PRNGKey(0), 5)

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
    
MCMC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
MCMC is out of the scope of this package, but we can integrate with other packages to
sample flowjax distributions with MCMC, for example using [numpyro](https://github.com/pyro-ppl/numpyro)

.. code-block:: python

    >>> from numpyro.infer import MCMC, NUTS
    >>> import numpyro.distributions as ndist
    >>> from numpyro.distributions import constraints
    >>> from numpyro import sample
    >>> import flowjax.distributions as fdist
    >>> import jax.random as jr
    >>> import numpy as np

    >>> class NumpyroWrapper(ndist.Distribution):
    ...     "Wraps a flowjax distribution into a numpyro distribution"

    ...     def __init__(self, distribution: fdist.Distribution, support=constraints.real):
    ...         self.distribution = distribution
    ...         self.support = support
    ...         super().__init__(batch_shape=(), event_shape=distribution.shape)
     
    >>>     def log_prob(self, value):
    ...         return self.distribution.log_prob(value)
    
    >>>     def sample(self, key, sample_shape=...):
    ...         return self.distribution.sample(key, sample_shape)


    >>> def numpyro_model(X, y):
    ...     "Example regression model defined in terms of flowjax distributions"
    ...     beta = sample("beta", NumpyroWrapper(fdist.Normal(np.zeros(2))))
    ...     sample("y", NumpyroWrapper(fdist.Normal(X @ beta)), obs=y)

    >>> X = np.random.randn(100, 2)
    >>> beta_true = np.array([-1, 1])
    >>> y = X @ beta_true + np.random.randn(100)
    >>> mcmc = MCMC(NUTS(numpyro_model), num_warmup=100, num_samples=1000)
    >>> mcmc.run(jr.PRNGKey(0), X, y)

This approach (or similar) may be useful if we want to use a flow as part of a 
model.
