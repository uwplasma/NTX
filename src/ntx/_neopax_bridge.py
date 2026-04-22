"""Differentiable NTX-to-NEOPAX array mapping helpers."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ._neopax_types import D33_MODES, NeopaxMonoenergeticArrays, NeopaxScan
from .geometry import BoozerSurface, VmecSurface


def scan_to_neopax_arrays(
    scan: NeopaxScan,
    *,
    a_b: float | Array,
    d33_mode: str = "spitzer",
) -> NeopaxMonoenergeticArrays:
    """Map NTX scan data into the pure arrays consumed by `NEOPAX.Monoenergetic`."""

    rho = jnp.asarray(scan.rho)
    nu_v = jnp.asarray(scan.nu_v)
    er = jnp.asarray(scan.Er)
    drds = jnp.asarray(scan.drds)
    d11 = jnp.asarray(scan.D11)
    d13 = jnp.asarray(scan.D13)
    if d33_mode not in D33_MODES:
        raise ValueError(f"d33_mode must be one of {sorted(D33_MODES)}")
    if d33_mode == "spitzer":
        d33 = (
            jnp.asarray(scan.D33_spitzer)
            if scan.D33_spitzer is not None
            else jnp.asarray(scan.D33)
        )
    elif d33_mode == "raw":
        d33 = jnp.asarray(scan.D33)
    else:
        if scan.D33_spitzer is None:
            raise ValueError(
                "d33_mode='conductivity_difference' requires D33_spitzer in the scan"
            )
        d33 = jnp.asarray(scan.D33_spitzer) - jnp.asarray(scan.D33)
    a_b_value = jnp.asarray(a_b)
    d13 = d13 * drds[:, None, None]

    er0 = er[0]
    er_list = jnp.stack(
        [
            jnp.log10(jnp.maximum(1.0e-8, jnp.abs(er0) / (a_b_value * rho_value)))
            for rho_value in rho
        ]
    )
    return NeopaxMonoenergeticArrays(
        a_b=a_b_value,
        rho=rho,
        nu_log=jnp.log10(nu_v),
        Er_list=er_list,
        D11_log=jnp.log10(d11 * drds[:, None, None] ** 2),
        D13=d13,
        D33=d33 * nu_v[None, :, None],
    )


def to_neopax_monoenergetic(
    scan: NeopaxScan,
    *,
    a_b: float,
    d33_mode: str = "spitzer",
):
    """Construct `NEOPAX.Monoenergetic` from NTX scan data."""

    try:
        import NEOPAX
    except ImportError as exc:  # pragma: no cover - exercised when NEOPAX exists locally
        raise ImportError("NEOPAX is required for `to_neopax_monoenergetic`") from exc

    arrays = scan_to_neopax_arrays(scan, a_b=a_b, d33_mode=d33_mode)

    return NEOPAX.Monoenergetic(
        a_b=float(a_b),
        rho=arrays.rho,
        nu_log=arrays.nu_log,
        Er_list=arrays.Er_list,
        D11_log=arrays.D11_log,
        D13=arrays.D13,
        D33=arrays.D33,
    )


def _surface_transport_scale(surface: BoozerSurface | VmecSurface) -> Array:
    if isinstance(surface, VmecSurface):
        return jnp.asarray(surface.transport_psi_scale, dtype=jnp.float64)
    return jnp.asarray(surface.psi_p, dtype=jnp.float64)


def _surface_reference_bridge(surface: BoozerSurface | VmecSurface) -> dict[str, Array]:
    if isinstance(surface, VmecSurface):
        zero_mode = jnp.asarray((surface.m == 0) & (surface.n == 0))
        idx = jnp.argmax(zero_mode.astype(jnp.int32))
        boozer_i = jnp.asarray(jnp.take(surface.b_sub_theta_cos, idx), dtype=jnp.float64)
        boozer_g = jnp.asarray(jnp.take(surface.b_sub_zeta_cos, idx), dtype=jnp.float64)
        psi_a = jnp.asarray(surface.psi_a_hat, dtype=jnp.float64)
        b00 = jnp.asarray(surface.b0, dtype=jnp.float64)
        iota = jnp.asarray(surface.iota, dtype=jnp.float64)
    else:
        boozer_i = jnp.asarray(surface.b_theta, dtype=jnp.float64)
        boozer_g = jnp.asarray(surface.b_zeta, dtype=jnp.float64)
        psi_a = jnp.asarray(surface.psi_p, dtype=jnp.float64)
        b00_source = surface.b0 if surface.b0 is not None else surface.b_cos[0]
        b00 = jnp.asarray(b00_source, dtype=jnp.float64)
        iota = jnp.asarray(surface.iota, dtype=jnp.float64)

    denom = boozer_g + iota * boozer_i
    fac_11 = 8.0 * denom * b00 * psi_a**2 / (jnp.sqrt(jnp.pi) * boozer_g**2)
    fac_31 = 4.0 * b00 * psi_a / (jnp.sqrt(jnp.pi) * boozer_g)
    fac_33 = 2.0 * b00 / (jnp.sqrt(jnp.pi) * denom)
    dpsi_drtilde = surface.r_hat * b00 if isinstance(surface, VmecSurface) else b00
    fac_sfincs_to_dkes_11 = 1.0 / (
        8.0 * denom * dpsi_drtilde**2 / (boozer_g**2 * b00 * jnp.sqrt(jnp.pi))
    )
    fac_sfincs_to_dkes_31 = 1.0 / (4.0 * dpsi_drtilde / (boozer_g * jnp.sqrt(jnp.pi)))
    fac_sfincs_to_dkes_33 = 1.0 / (2.0 * b00 / (denom * jnp.sqrt(jnp.pi)))
    return {
        "b00": b00,
        "boozer_i": boozer_i,
        "boozer_g": boozer_g,
        "iota": iota,
        "fac_11": fac_11,
        "fac_31": fac_31,
        "fac_33": fac_33,
        "fac_sfincs_to_dkes_11": fac_sfincs_to_dkes_11,
        "fac_sfincs_to_dkes_31": fac_sfincs_to_dkes_31,
        "fac_sfincs_to_dkes_33": fac_sfincs_to_dkes_33,
    }
