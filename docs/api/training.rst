Training
==========================
FlowJax includes a training script for convenience, if users want to fit a flow via maximum likelihood
using samples from the target density (and corresponding conditioning variables if appropriate).

.. autofunction:: flowjax.train.fit_to_data

We also provide a function for performing variational inference with a flow. See the examples for 
illustrations of how to use this function.

.. autofunction:: flowjax.train.fit_to_variational_target