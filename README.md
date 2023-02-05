<div align="center">
<img src="./images/flowjax_logo.png?raw=true" alt="logo" width="500" ></img>
</div>

# FlowJax: Normalising Flows in Jax

## Documentation
Available [here](https://danielward27.github.io/flowjax/index.html).

## Short example
Training a flow can be done in a few lines of code:

```
from flowjax.flows import BlockNeuralAutoregressiveFlow
from flowjax.train import fit_to_data
from flowjax.distributions import Normal
from jax import random
import jax.numpy as jnp

data_key, flow_key, train_key = random.split(random.PRNGKey(0), 3)

x = random.uniform(data_key, (10000, 3))  # Toy data
base_dist = Normal(jnp.zeros(x.shape[1]))
flow = BlockNeuralAutoregressiveFlow(flow_key, base_dist)
flow, losses = fit_to_data(train_key, flow, x, learning_rate=0.05)

# We can now evaluate the log-probability of arbitrary points
flow.log_prob(x)
```

The package currently supports the following:

- `CouplingFlow` ([Dinh et al., 2017](https://arxiv.org/abs/1605.08803))
- `MaskedAutoregressiveFlow` ([Papamakarios et al., 2017](https://arxiv.org/abs/1705.07057v4)).
- Common "transformers", such as `Affine` and `RationalQuadraticSpline` (the latter used in neural spline flows; [Durkan et al., 2019](https://arxiv.org/abs/1906.04032))
- `BlockNeuralAutoregressiveFlow`, as introduced by [De Cao et al., 2019](https://arxiv.org/abs/1904.04676)
- `TriangularSplineFlow`, introduced here.


## Installation
```
pip install flowjax
```

## Warning
This package is new and may have substantial breaking changes between major releases.

## TODO
A few limitations / things that could be worth including in the future:
- Add ability to "reshape" bijections.
- Ability to "stack" bijections on a given (or new) axis.

## Related
We make use of the [Equinox](https://arxiv.org/abs/2111.00254) package, which facilitates object-oriented programming with Jax. 

## Authors
`flowjax` was written by `Daniel Ward <danielward27@outlook.com>`.

