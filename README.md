
![FlowJAX](/docs/_static/logo_light.svg)

Distributions, bijections and normalizing flows using Equinox and JAX
-----------------------------------------------------------------------
- Includes a wide range of distributions and bijections.
- Distributions and bijections are PyTrees, registered through 
  [Equinox](https://github.com/patrick-kidger/equinox/) modules, making them
  compatible with [JAX](https://github.com/google/jax) transformations.
- Includes many state of the art normalizing flow models.
- First class support for conditional distributions and density estimation.

## Documentation
Available [here](https://danielward27.github.io/flowjax/index.html).

## Short example
As an example we will create and train a normalizing flow model to toy data in just a few lines of code:

```python
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.distributions import Normal
import jax.random as jr
import jax.numpy as jnp

data_key, flow_key, train_key, sample_key = jr.split(jr.key(0), 4)

x = jr.uniform(data_key, (5000, 2))  # Toy data

flow = block_neural_autoregressive_flow(
    key=flow_key,
    base_dist=Normal(jnp.zeros(x.shape[1])),
)

flow, losses = fit_to_data(
    key=train_key,
    dist=flow,
    x=x,
    learning_rate=5e-3,
    max_epochs=200,
    )

# We can now evaluate the log-probability of arbitrary points
log_probs = flow.log_prob(x)

# And sample the distribution
samples = flow.sample(sample_key, (1000, ))
```

The package currently includes:
- Many simple bijections and distributions, implemented as [Equinox](https://arxiv.org/abs/2111.00254) modules.
- `coupling_flow` ([Dinh et al., 2017](https://arxiv.org/abs/1605.08803)) and `masked_autoregressive_flow` ([Kingma et al., 2016](https://arxiv.org/abs/1606.04934), [Papamakarios et al., 2017](https://arxiv.org/abs/1705.07057v4)) normalizing flow architectures.
    - These can be used with arbitrary bijections as transformers, such as `Affine` or `RationalQuadraticSpline` (the latter used in neural spline flows; [Durkan et al., 2019](https://arxiv.org/abs/1906.04032)). 
- `block_neural_autoregressive_flow`, as introduced by [De Cao et al., 2019](https://arxiv.org/abs/1904.04676).
- `planar_flow`, as introduced by [Rezende and Mohamed, 2015](https://arxiv.org/pdf/1505.05770.pdf).
- `triangular_spline_flow`, introduced here.
- Training scripts for fitting by maximum likelihood, variational inference, or using contrastive learning for sequential neural posterior estimation ([Greenberg et al., 2019](https://arxiv.org/abs/1905.07488); [Durkan et al., 2020](https://arxiv.org/abs/2002.03712])).
- A bisection search algorithm that allows inverting some bijections without a
known inverse, allowing for example both sampling and density evaluation to be
performed with block neural autoregressive flows.

## Installation
```bash
pip install flowjax
```

## Warning
This package is in its early stages of development and may undergo significant changes, including breaking changes, between major releases. Whilst ideally we should be on version 0.y.z to indicate its state, we have already progressed beyond that stage. Any breaking changes will be in the release notes for each major release.

## Development
We can install a version for development as follows
```bash
git clone https://github.com/danielward27/flowjax.git
cd flowjax
pip install -e .[dev]
sudo apt-get install pandoc  # Required for building documentation
```

## Related
We make use of the [Equinox](https://arxiv.org/abs/2111.00254) package, which
facilitates defining models using a PyTorch-like syntax with Jax. 

## Citation
If you found this package useful in academic work, please consider citing it using the
template below, filling in ``[version number]`` and ``[release year of version]`` to the
appropriate values. Version specific DOIs
can be obtained from [zenodo](https://zenodo.org/records/10402073) if desired.

```bibtex
@software{ward2023flowjax,
  title = {FlowJAX: Distributions and Normalizing Flows in Jax},
  author = {Daniel Ward},
  url = {https://github.com/danielward27/flowjax},
  version = {[version number]},
  year = {[release year of version]},
  doi = {10.5281/zenodo.10402073},
}
```
