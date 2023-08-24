Training
==========================
FlowJax includes basic training scripts for convenience, although users may need to modify these
for specific use cases. If we wish to fit the flow to samples from a distribution (and
corresponding conditioning variables if appropriate), we can use ``fit_to_data``.

.. autofunction:: flowjax.train.fit_to_data

Alternatively, we can also  provide a function for performing variational inference with a flow. See the examples for 
illustrations of how to use this function.

.. autofunction:: flowjax.train.fit_to_variational_target