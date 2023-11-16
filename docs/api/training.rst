Training
==========================
FlowJax includes basic training scripts for convenience, although users may need to modify these
for specific use cases. If we wish to fit the flow to samples from a distribution (and
corresponding conditioning variables if appropriate), we can use ``fit_to_data``.

.. autofunction:: flowjax.train.fit_to_data

Alternatively, we can use ``fit_to_variational_target`` to fit the flow to a function
using variational inference.

.. autofunction:: flowjax.train.fit_to_variational_target
