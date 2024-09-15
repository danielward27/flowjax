
Introduction to distributions
---------------------------------

As an example, we will use :func:`flowja.distributions.Normal`. In general, for simple
distributoins the parameters are broadcast, and used to infer the shape of the 
distribution.

.. doctest:: 
   
   >>> from flowjax.distributions import Normal
   >>> import jax.numpy as jnp
   >>> normal = Normal(loc=0, scale=jnp.ones(3))
   >>> normal.shape
   (3,)

We can sample the distribution, either a single value with shape ``normal.shape``,
or a batch

.. doctest:: 
   
   >>> import jax.random as jr
   >>> key = jr.key(0)
   >>> sample = normal.sample(key)
   >>> sample.shape
   (3,)
   >>> batch = normal.sample(key, (4, ))
   >>> batch.shape
   (4, 3)

and we can evaluate the log probabilities of these samples

.. doctest:: 
   
   >>> import jax.random as jr
   >>> normal.log_prob(sample)
   Array(-3.4016984, dtype=float32)
   >>> normal.log_prob(batch)
   Array([-4.8808994, -5.0121717, -3.2557464, -4.131773 ], dtype=float32)

where the output shape of the log probabilities matches the batch shape.
TODO lack of batch shape...

In general, the underlying parameterizations are constrained in a sensible manner,
for example, using softplus to enforce positivity of scale parameters. In practice,
means you can optimize over the distributions of parameters without concern for invalid
values.

Introduction