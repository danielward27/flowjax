## flowjax
-------

Normalising flow implementations in jax. Training a flow can be done in a few lines of code

```
from flowjax.flows import BlockNeuralAutoregressiveFlow
from flowjax.train_utils import train_flow
from flowjax.distributions import Normal
from jax import random

data_key, flow_key, train_key = random.split(random.PRNGKey(0), 3)

x = random.uniform(data_key, (10000, 3))  # Toy data
flow = BlockNeuralAutoregressiveFlow(flow_key, Normal(3))
flow, losses = train_flow(train_key, flow, x, learning_rate=0.05)

# We can now evaluate the log-probability of arbitrary points
flow.log_prob(x)
```

So far the package supports the following:

- Affine coupling flows/RealNVP for conditional/unconditional density estimation and sampling ([Dinh *et al.*](https://arxiv.org/abs/1605.08803))

- Neural spline coupling flows for conditional/unconditional density estimation and sampling ([Durkan *et al.*](https://arxiv.org/abs/1906.04032/))

- Block neural autoregressive flows for conditional/unconditional density estimation ([De Cao *et al.*](https://arxiv.org/abs/1904.04676))


For more detailed examples, see [examples](https://github.com/danielward27/flowjax/blob/main/examples/).

## Installation
```
pip install flowjax
```

## Authors
`flowjax` was written by `Daniel Ward <danielward27@outlook.com>`.
