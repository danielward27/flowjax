# We do this for now due to an incompatibility between equinox abstract class
# extensions and the documentation generator sphinx
# https://github.com/patrick-kidger/equinox/issues/591. This will likely be fixable with
# https://peps.python.org/pep-0649/ in python 3.13
import builtins

if getattr(builtins, "GENERATING_DOCUMENTATION", False):

    class ArrayLike:
        pass

else:
    from jaxtyping import ArrayLike  # noqa: F401
