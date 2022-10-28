## flowjax
-------

Normalising flows in JAX. Training a flow can be done in a few lines of code:

```
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.train_utils import train_flow
from flowjax.distributions import Normal
from jax import random
import jax.numpy as jnp

data_key, flow_key, train_key = random.split(random.PRNGKey(0), 3)

x = random.uniform(data_key, (10000, 3))  # Toy data
base_dist = Normal(jnp.zeros(x.shape[1]))
flow = block_neural_autoregressive_flow(flow_key, base_dist)
flow, losses = train_flow(train_key, flow, x, learning_rate=0.05)

# We can now evaluate the log-probability of arbitrary points
flow.log_prob(x)
```

The package currently supports the following:

- Supports both `coupling_flow` ([Dinh et al., 2017](https://arxiv.org/abs/1605.08803)) and `masked_autoregressive_flow` ([Papamakarios et al., 2017](https://arxiv.org/abs/1705.07057v4))  architectures
- Supports common transformers, such as `AffineTransformer` and `RationalQuadraticSplineTransformer` (the latter used in neural spline flows; [Durkan et al., 2019](https://arxiv.org/abs/1906.04032))
- `block_neural_autoregressive_flow`, as introduced by [De Cao et al., 2019](https://arxiv.org/abs/1904.04676)

For examples of basic usage, see [examples](https://github.com/danielward27/flowjax/blob/main/examples/).

## Installation
```
pip install flowjax
```

## Warning
This package is new and may have substantial breaking changes between major releases.

## TODO
A few limitations / things that could be worth including in the future:

- Support embedding networks (for dimensionality reduction of conditioning variables)
- Add batch/layer normalisation to neural networks
- Training script for variational inference
- Add documentation

## Related
We make use of the [Equinox](https://arxiv.org/abs/2111.00254) package, which facilitates object-oriented programming with Jax. 

## FAQ
**How to avoid training the base distribution?**
Provide a `filter_spec` to `train_flow`, for example
```
import equinox as eqx
import jax.tree_util as jtu
filter_spec = jtu.tree_map(lambda x: eqx.is_inexact_array(x), flow)
filter_spec = eqx.tree_at(lambda tree: tree.base_dist, filter_spec, replace=False)
```

**Do I need to scale my variables?**
In general yes, you should consider the form and scales of the target samples. Often it is useful to define a bijection to carry out the preprocessing, then to transform the flow with the inverse, to "undo" the preprocessing. For example, to carry out "standard scaling", we could do
```
import jax
from flowjax.bijections import Affine, Invert
from flowjax.distributions import Transformed

preprocess = Affine(-x.mean(axis=0)/x.std(axis=0), 1/x.std(axis=0))
x_processed = jax.vmap(preprocess.transform)(x)
flow, losses = train_flow(train_key, flow, x_processed)
flow = Transformed(flow, Invert(preprocess))  # "undo" the preprocessing
```

## Authors
`flowjax` was written by `Daniel Ward <danielward27@outlook.com>`.

