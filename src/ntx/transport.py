"""Transport coefficient post-processing."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .geometry import GeometryOnGrid
from .grids import unflatten_fs


def coefficients_from_modes(
    geom: GeometryOnGrid,
    psi_p: float,
    f1_modes: Array,
    f3_modes: Array,
    nu_hat: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Compute `(D11, D31, D13, D33, D33_spitzer)` from modes 0, 1, and 2."""

    n_theta, n_zeta = geom.b.shape
    f10 = unflatten_fs(f1_modes[0], n_theta, n_zeta)
    f11 = unflatten_fs(f1_modes[1], n_theta, n_zeta)
    f12 = unflatten_fs(f1_modes[2], n_theta, n_zeta)
    f30 = unflatten_fs(f3_modes[0], n_theta, n_zeta)
    f31 = unflatten_fs(f3_modes[1], n_theta, n_zeta)
    f32 = unflatten_fs(f3_modes[2], n_theta, n_zeta)

    vm0 = geom.radial_drift_spatial * (2.0 / 3.0)
    vm2 = geom.radial_drift_spatial / 3.0
    pref = geom.jacobian / geom.volume_prime
    dtheta_dzeta = geom.grid.dtheta * geom.grid.dzeta

    def avg(value: Array) -> Array:
        return jnp.sum(pref * value) * dtheta_dzeta

    d11 = avg(-2.0 * vm0 * f10 - 2.0 * vm2 * f12 / 5.0) / psi_p**2
    d31 = avg(2.0 * f11 * geom.b / (3.0 * geom.b0)) / psi_p
    d13 = avg(-2.0 * vm0 * f30 - 2.0 * vm2 * f32 / 5.0) / (psi_p * geom.b0)
    d33 = avg(2.0 * geom.b * f31 / (3.0 * geom.b0)) / geom.b0
    d33_spitzer = avg(2.0 * geom.b * (geom.b / (geom.b0 * nu_hat)) / 3.0) / geom.b0
    return d11, d31, d13, d33, d33_spitzer


def onsager_error(d31: Array, d13: Array) -> Array:
    """Return `|D13 + D31|`, the relevant monoenergetic Onsager residual."""

    return jnp.abs(d13 + d31)
