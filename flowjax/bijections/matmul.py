"""Affine bijections."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from jax import Array

from flowjax.bijections.bijection import AbstractBijection


class MatMul(AbstractBijection):
    """Matrix multiplication transformation ``y = A @ x``.

    This is useful for whitening data by a (inverse-)covariance matrix.

    Args:
        mat: square matrix.
        inv_mat: square matrix | None
            If `None`, the inverse matrix is computed using `jnp.linalg.inv`.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    mat: Array
    inv_mat: Array

    def __init__(self, mat: Array, inv_mat: Array | None = None) -> None:
        self.mat = mat
        self.inv_mat = jnp.linalg.inv(mat) if inv_mat is None else inv_mat
        self.shape = mat.shape[:1]

    def transform(self, x, condition=None):
        return self.mat @ x

    def transform_and_log_det(self, x, condition=None):
        y = self.mat @ x
        return y, jnp.linalg.slogdet(self.mat)[1]

    def inverse(self, y, condition=None):
        return self.inv_mat @ y

    def inverse_and_log_det(self, y, condition=None):
        x = self.inv_mat @ y
        return x, jnp.linalg.slogdet(self.inv_mat)[1]
