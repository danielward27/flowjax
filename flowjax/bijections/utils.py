from flowjax.bijections.abc import Bijection
import jax.numpy as jnp
import equinox as eqx


class Chain(Bijection, eqx.Module):
    bijections: list[Bijection]
    # TODO add options for permute/flip?

    def __init__(self, bijections: list):
        """Chain together bijections to form another bijection.

        Args:
            bijections (list): List of bijections.
        """
        self.bijections = bijections

    def transform(self, x, condition=jnp.array([])):
        z = x
        for bijection in self.bijections:
            z = bijection.transform(z, condition)
        return z

    def transform_and_log_abs_det_jacobian(self, x, condition=jnp.array([])):
        log_abs_det_jac = 0
        z = x
        for bijection in self.bijections:
            z, log_abs_det_jac_i = bijection.transform_and_log_abs_det_jacobian(
                z, condition
            )
            log_abs_det_jac += log_abs_det_jac_i
        return z, log_abs_det_jac

    def inverse(self, z, condition=jnp.array([])):
        x = z
        for bijection in reversed(self.bijections):
            x = bijection.inverse(x, condition)
        return x


class Permute(Bijection):
    permutation: jnp.ndarray  # with indices
    inverse_permutation: jnp.ndarray

    def __init__(self, permutation):
        """Permutation transformation. Note condition is ignored.

        Args:
            permutation (jnp.ndarray): Indexes 0-(dim-1) representing new order.
        """
        assert (permutation.sort() == jnp.arange(len(permutation))).all()
        self.permutation = permutation
        self.inverse_permutation = jnp.argsort(permutation)

    def transform(self, x, condition=jnp.array([])):
        return x[self.permutation]

    def transform_and_log_abs_det_jacobian(self, x, condition=jnp.array([])):
        return x[self.permutation], jnp.array(0)

    def inverse(self, y, condition=jnp.array([])):
        return y[self.inverse_permutation]

