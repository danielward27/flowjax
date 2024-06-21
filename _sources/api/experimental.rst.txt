Experimental
==========================

Interfacing with numpyro
--------------------------

Supporting complex inference approaches such as MCMC or variational inference
with arbitrary probabilistic models is out of the scope of this package. However,
we do provide some basic suppport for interfacing with numpyro. We note this support is
in its infancy and there may be breaking changes without warning. 

.. warning::
    Batch dimensions are handled differently for flowjax distributions and numpyro
    distributions. In flowjax we do not make a clear distinction between
    event shapes and batch shapes. Hence, when a flowjax distribution is converted to a
    numpyro distribution, we assume its shape corresponds to the event shape.
    
In general, we can use a combination of flowjax and numpyro distributions in a
numpyro model by using :func:`~flowjax.experimental.numpyro.sample`, in place of
numpyro's ``sample``. This will wrap flowjax distributions to numpyro
distributions, using :func:`~flowjax.experimental.numpyro.distribution_to_numpyro`.
This approach can be used for example to embed normalising flows into arbitrary
probabilistic models. Here is a simple example

    .. doctest::


        >>> from numpyro.infer import MCMC, NUTS
        >>> from flowjax.experimental.numpyro import sample
        >>> from flowjax.distributions import Normal
        >>> import jax.random as jr
        >>> import numpy as np

        >>> def numpyro_model(X, y):
        ...     "Example regression model defined in terms of flowjax distributions"
        ...     beta = sample("beta", Normal(np.zeros(2)))
        ...     sample("y", Normal(X @ beta), obs=y)

        >>> X = np.random.randn(100, 2)
        >>> beta_true = np.array([-1, 1])
        >>> y = X @ beta_true + np.random.randn(100)
        >>> mcmc = MCMC(NUTS(numpyro_model), num_warmup=10, num_samples=100)
        >>> mcmc.run(jr.PRNGKey(0), X, y)

.. automodule:: flowjax.experimental.numpyro
   :members:
   :undoc-members:
   :show-inheritance:
