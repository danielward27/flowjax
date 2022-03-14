from flowjax.bijections.abc import Bijection
import jax.numpy as jnp


class Permute(Bijection):
    permutation: jnp.ndarray  # with indices
    inverse_permutation: jnp.ndarray

    def __init__(self, permutation):
        """Permutation transformation.

        Args:
            permutation (jnp.ndarray): Indexes 0-(dim-1) representing new order.
        """
        assert (permutation.sort() == jnp.arange(len(permutation))).all()
        self.permutation = permutation
        self.inverse_permutation = jnp.argsort(permutation)

    def transform(self, x):
        return x[self.permutation]

    def transform_and_log_abs_det_jacobian(self, x):
        return x[self.permutation], jnp.array(0)

    def inverse(self, y):
        return y[self.inverse_permutation]
