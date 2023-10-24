Experimental
==========================

Interfacing with numpyro
--------------------------

Supporting complex inference approaches such as MCMC or variational inference
with arbitrary probabilistic models is out of the scope of this package. However, we do
provide an (experimental) wrapper class,
:class:`~flowjax.experimental.numpyro.TransformedToNumpyro`, which will wrap
a flowjax :class:`~flowjax.distributions.AbstractTransformed` distribution, into a 
`numpyro <https://github.com/pyro-ppl/numpyro>`_ distribution.
This can be used for example to embed normalising flows into arbitrary
probabilistic models. Here is a simple example

    .. doctest::

        >>> from numpyro.infer import MCMC, NUTS
        >>> from flowjax.experimental.numpyro import TransformedToNumpyro
        >>> from numpyro import sample
        >>> from flowjax.distributions import Normal
        >>> import jax.random as jr
        >>> import numpy as np

        >>> def numpyro_model(X, y):
        ...     "Example regression model defined in terms of flowjax distributions"
        ...     beta = sample("beta", TransformedToNumpyro(Normal(np.zeros(2))))
        ...     sample("y", TransformedToNumpyro(Normal(X @ beta)), obs=y)

        >>> X = np.random.randn(100, 2)
        >>> beta_true = np.array([-1, 1])
        >>> y = X @ beta_true + np.random.randn(100)
        >>> mcmc = MCMC(NUTS(numpyro_model), num_warmup=10, num_samples=100)
        >>> mcmc.run(jr.PRNGKey(0), X, y)

.. automodule:: flowjax.experimental.numpyro
   :members:
   :undoc-members:
   :show-inheritance:
