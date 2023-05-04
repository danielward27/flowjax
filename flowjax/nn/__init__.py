"""nn package contains neural network architectures of use for flows (invertible
neural networks).
"""

from .block_autoregressive import BlockAutoregressiveLinear, _block_tanh_activation
from .masked_autoregressive import AutoregressiveMLP, MaskedLinear

__all__ = [
    "MaskedLinear",
    "AutoregressiveMLP",
    "BlockAutoregressiveLinear",
    "_block_tanh_activation",
]
