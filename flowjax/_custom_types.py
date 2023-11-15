# ArrayLike here is just an alias for Any instead of jax.typing.ArrayLike
# We do this for now due to an incompatibility between equinox abstract class
# extensions and the documentation generator sphinx
# https://github.com/patrick-kidger/equinox/issues/591. This will likely be fixable with
#  https://peps.python.org/pep-0649/ in python 3.13
from typing import Any

ArrayLike = Any
