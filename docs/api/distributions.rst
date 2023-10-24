Distributions
==========================
Distributions from ``flowjax.distributions``.

.. automodule:: flowjax.distributions
   :members:
   :show-inheritance:
   :member-order: groupwise

Implementing a custom distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To implement a custom distribution, subclass
    -  :class:`~flowjax.distributions.AbstractStandardDistribution` for non-transformed distributions.
    -  :class:`~flowjax.distributions.AbstractTransformed` for transformed distributions.

For simple examples, check the source code for
   - :class:`~flowjax.distributions.StandardNormal` as an example of an :class:`~flowjax.distributions.AbstractStandardDistribution`.
   - :class:`~flowjax.distributions.Normal` as an example of an :class:`~flowjax.distributions.AbstractTransformed`.

