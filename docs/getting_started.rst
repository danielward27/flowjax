Getting Started
-----------------
If you are interested in use of normalizing flows, feel free to start with an example
on the left side page. This page gives an overview of FlowJAX distributions and
bijections more generally.

Simple Distribution Example
============================

In this example, we will use :func:`~flowjax.distributions.Normal` to demonstrate the
behavior of FlowJAX distributions.

.. doctest:: 
   
   >>> from flowjax.distributions import Normal
   >>> import jax.numpy as jnp
   >>> import jax.random as jr
   >>> normal = Normal(loc=jnp.arange(3), scale=1)
   >>> normal.shape
   (3,)

We can sample from the distribution, either as a single value or a batch of independent and identically distributed (iid) samples:

.. doctest:: 
   
   >>> key = jr.key(0)
   >>> sample = normal.sample(key)
   >>> sample.shape
   (3,)
   >>> batch = normal.sample(key, (4,))
   >>> batch.shape
   (4, 3)

Additionally, we can evaluate the log probabilities of these samples:

.. doctest:: 
   
   >>> normal.log_prob(sample)
   Array(-3.4016984, dtype=float32)
   >>> normal.log_prob(batch)
   Array([-4.8808994, -5.0121717, -3.2557464, -4.131773 ], dtype=float32)

When ``sample.shape == distribution.shape``, a scalar log probability is returned. For 
a batch of samples, the shape of the returned log probabilities matches the shape
of the extra leading dimensions.

Conditional Distribution Example
=================================

FlowJAX also supports conditional distributions. All distributions have a ``cond_shape``
attribute, which is ``None`` for unconditional distributions. For conditional 
distributions, this attribute is a tuple representing the shape of the conditioning variable.

As an example, we construct a :func:`~flowjax.flows.coupling_flow`:

.. doctest::

   >>> from flowjax.flows import coupling_flow
   >>> dist = coupling_flow(key, base_dist=Normal(jnp.zeros(3)), cond_dim=2)
   >>> dist.shape
   (3,)
   >>> dist.cond_shape
   (2,)

The distribution methods follow NumPy's broadcasting rules. The output shape for
sampling is ``sample_shape + condition_batch_shape + dist.shape``, while the log
probability shape is ``sample_shape + condition_batch_shape``. For example:

.. doctest ::

   >>> # Sampling 10 times for a single conditioning variable instance
   >>> condition = jnp.ones(2)
   >>> samples = dist.sample(key, (10,), condition=condition)
   >>> samples.shape
   (10, 3)
   >>> dist.log_prob(samples, condition).shape
   (10,)
   >>> # Sampling once for each of 5 conditioning variable instances.
   >>> condition = jnp.ones((5, 2))
   >>> samples = dist.sample(key, condition=condition)
   >>> samples.shape
   (5, 3)
   >>> dist.log_prob(samples, condition).shape
   (5,)

Bijections
==========

Bijections are invertible, differentiable transformations that can be used to
transform distributions. For instance, :py:class:`~flowjax.bijections.Affine` performs the transformation
:math:`y = a \cdot x + b`:

.. doctest::

   >>> import jax.numpy as jnp
   >>> from flowjax.bijections import Affine
   >>> bijection = Affine(loc=0, scale=2)
   >>> x = 1
   >>> y = bijection.transform(x)  # shapes must match!
   >>> y
   Array(2., dtype=float32)  
   >>> bijection.inverse(y)  # shapes must match!
   Array(1., dtype=float32)

You can also compute the log determinant alongside the forward or inverse transformation:

.. doctest:: 

   >>> bijection.transform_and_log_det(x)
   (Array(2., dtype=float32), Array(0.6931472, dtype=float32))
   >>> bijection.inverse_and_log_det(y)
   (Array(1., dtype=float32), Array(-0.6931472, dtype=float32))

Similar to distributions, bijections can be conditional or unconditional, and they have 
``shape`` and ``cond_shape`` attributes. The latter is ``None`` for unconditional
bijections. Unlike distributions, array shapes must match exactly—no automatic broadcasting.
To vectorize over bijection methods, it may be useful to apply ``jax.vmap``:

.. doctest:: 

   >>> import jax
   >>> from flowjax.bijections import Scale
   >>> scale = Scale(2)  # shape ()
   >>> x = jnp.arange(3)
   >>> jax.vmap(scale.transform)(x)
   Array([0., 2., 4.], dtype=float32)

Transforming Distributions
==========================

FlowJAX provides two methods for defining transformed distributions. We'll create a log-normal distribution using both approaches.

**Option 1**: Using :py:class:`~flowjax.distributions.Transformed` which takes a base
distribution and a transformation (bijection) as arguments:

.. doctest::

   >>> from jaxtyping import ArrayLike
   >>> from flowjax.distributions import Normal, Transformed
   >>> from flowjax.bijections import Exp
   >>> log_normal = Transformed(Normal(), Exp())

**Option 2**: Inheriting from :py:class:`~flowjax.distributions.AbstractTransformed`.
This approach is more flexible for example if you wish to add extra methods or
attributes.

.. doctest::

   >>> from flowjax.distributions import Normal, AbstractTransformed
   >>> from flowjax.bijections import Exp
   >>> class LogNormal(AbstractTransformed):
   ...     base_dist: Normal
   ...     bijection: Exp
   ...
   ...     def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
   ...         self.base_dist = Normal(loc, scale)
   ...         self.bijection = Exp(self.base_dist.shape)
   ...
   >>> log_normal = LogNormal()

.. note:: 
   In either case, the ``shapes`` must match. Further, you can arbitrarily combine
   unconditional and conditional bijections with unconditional and conditional
   distributions, as long as all conditional components share the same ``cond_shape``.

Distributions and Bijections as PyTrees
=======================================

Distributions and bijections are PyTrees, registered through
`equinox <https://github.com/patrick-kidger/equinox/>`_ modules. This allows us to 
use JAX/equinox operations on them. For instance, to define a batch of independent but
non-identically distributed distributions, we can vectorize the initialization:

.. doctest:: 
   
   >>> import equinox as eqx
   >>> normals = eqx.filter_vmap(Normal)(jnp.arange(3))  # batch of normals with shape ()
   >>> normals.shape
   ()

We can then vectorize the log probability computation over these parameters:

.. doctest:: 

   >>> log_probs = eqx.filter_vmap(lambda dist, x: dist.log_prob(x))(normals, jnp.arange(3))
   >>> log_probs.shape  # not scalar!
   (3,)

This approach avoids the need for a seperately specificying e.g. a ``batch_shape``,
which is often inconsistently available in other distribution packages.

Additional Notes
==================

- The underlying parameterizations are constrained for direct optimization 
  (e.g., positivity constraints for scale parameters).
- FlowJAX assumes optimization over inexact JAX arrays (complex or floating point arrays).
