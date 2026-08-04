"""Periodic angular grids and spectral derivative matrices."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, tree_util


@dataclass(frozen=True)
class GridSpec:
    """Resolution and dtype settings for a monoenergetic solve."""

    n_theta: int
    n_zeta: int
    n_xi: int
    dtype: str = "float64"
    x64: bool = True

    def __post_init__(self) -> None:
        """Reject grids too coarse to resolve a Fourier derivative.

    Three points is the minimum for which the spectral derivative is meaningful
    rather than merely computable.
        """
        if self.n_theta < 3 or self.n_zeta < 3:
            msg = "n_theta and n_zeta must be at least 3"
            raise ValueError(msg)
        if self.n_xi < 2:
            msg = "n_xi must be at least 2 so source modes 0, 1, and 2 are represented"
            raise ValueError(msg)

    @property
    def jax_dtype(self):
        """The JAX dtype this grid asks for, as a dtype rather than a name."""
        return jnp.float64 if self.dtype == "float64" else jnp.float32

    @property
    def n_fs(self) -> int:
        """Number of flux-surface points, ``n_theta * n_zeta``."""
        return self.n_theta * self.n_zeta


tree_util.register_dataclass(
    GridSpec,
    data_fields=(),
    meta_fields=("n_theta", "n_zeta", "n_xi", "dtype", "x64"),
)


@dataclass(frozen=True)
class AngularGrid:
    """Periodic theta/zeta grid with its spacings and spectral derivative matrices.

    Built once per :class:`GridSpec` and reused: the derivative matrices are the
    expensive part and do not depend on the case being solved.
    """
    theta: Array
    zeta: Array
    dtheta: Array
    dzeta: Array
    dtheta_matrix: Array
    dzeta_matrix: Array
    nfp: int


tree_util.register_dataclass(
    AngularGrid,
    data_fields=(
        "theta",
        "zeta",
        "dtheta",
        "dzeta",
        "dtheta_matrix",
        "dzeta_matrix",
    ),
    meta_fields=("nfp",),
)


def periodic_grid(spec: GridSpec, nfp: int) -> AngularGrid:
    """Create equispaced angular grids over one toroidal field period."""

    dtype = spec.jax_dtype
    theta = jnp.arange(spec.n_theta, dtype=dtype) * (2.0 * jnp.pi / spec.n_theta)
    zeta = jnp.arange(spec.n_zeta, dtype=dtype) * (2.0 * jnp.pi / (nfp * spec.n_zeta))
    dtheta = jnp.asarray(2.0 * jnp.pi / spec.n_theta, dtype=dtype)
    dzeta = jnp.asarray(2.0 * jnp.pi / (nfp * spec.n_zeta), dtype=dtype)
    return AngularGrid(
        theta=theta,
        zeta=zeta,
        dtheta=dtheta,
        dzeta=dzeta,
        dtheta_matrix=fourier_derivative_matrix(spec.n_theta, 2.0 * jnp.pi, dtype),
        dzeta_matrix=fourier_derivative_matrix(spec.n_zeta, 2.0 * jnp.pi / nfp, dtype),
        nfp=nfp,
    )


def fourier_derivative_matrix(n: int, period: float | Array, dtype=jnp.float64) -> Array:
    """Return the first-derivative Fourier collocation matrix.

    This implementation works for even or odd `n` and is built by differentiating
    the discrete Fourier basis, which keeps the formula compact and easy to test.
    """

    k = jnp.fft.fftfreq(n, d=float(period) / n) * (2.0 * jnp.pi)
    eye = jnp.eye(n, dtype=jnp.complex128 if dtype == jnp.float64 else jnp.complex64)

    def differentiate(column):
        return jnp.fft.ifft(1j * k * jnp.fft.fft(column))

    matrix = jnp.stack([differentiate(eye[:, i]) for i in range(n)], axis=1)
    return jnp.real(matrix).astype(dtype)


def flatten_fs(values: Array) -> Array:
    """Flatten a `(n_theta, n_zeta)` field using theta as the fastest index."""

    return jnp.ravel(values, order="F")


def unflatten_fs(values: Array, n_theta: int, n_zeta: int) -> Array:
    """Unflatten a flux-surface vector to `(n_theta, n_zeta)`."""

    return jnp.reshape(values, (n_theta, n_zeta), order="F")


def flux_surface_integral(values: Array, jacobian: Array, dtheta: Array, dzeta: Array) -> Array:
    """Compute the unnormalized flux-surface integral."""

    return jnp.sum(values * jacobian) * dtheta * dzeta


def flux_surface_average(values: Array, jacobian: Array, dtheta: Array, dzeta: Array) -> Array:
    """Compute a trapezoidal-rule flux-surface average."""

    volume_prime = flux_surface_integral(jnp.ones_like(values), jacobian, dtheta, dzeta)
    return flux_surface_integral(values, jacobian, dtheta, dzeta) / volume_prime
