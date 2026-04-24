"""JAX-safe NEOPAX field builders for imported VMEC/Boozer workflows."""

from __future__ import annotations

import interpax
import jax
import jax.numpy as jnp

from ._neopax_field_utils import (
    _find_mode_index,
    _safe_divide,
    _safe_reciprocal,
    _surface_b10,
    _surface_bsqav,
)
from ._neopax_types import DifferentiableNeopaxField
from ._neopax_vmec_jax_field import (
    _apply_boozer_sign_convention_profiles,
    _booz_xform_bundle_with_gmnc_from_vmec_jax_state,
    _booz_xform_gmnc_from_inputs,
    _rho_half_mesh_from_s,
    _vmec_edge_r00_from_state,
    _vmec_psia_from_indata,
    _vmec_psia_from_state,
    _vmec_volume_profiles_from_state,
    build_differentiable_neopax_field_from_vmec_jax_boundary_params,
    build_differentiable_neopax_field_from_vmec_jax_state,
)

__all__ = [
    "build_differentiable_neopax_field",
    "build_differentiable_neopax_field_from_vmec_jax_boundary_params",
    "build_differentiable_neopax_field_from_vmec_jax_state",
    "_apply_boozer_sign_convention_profiles",
    "_booz_xform_bundle_with_gmnc_from_vmec_jax_state",
    "_booz_xform_gmnc_from_inputs",
    "_find_mode_index",
    "_rho_half_mesh_from_s",
    "_safe_divide",
    "_safe_reciprocal",
    "_surface_b10",
    "_surface_bsqav",
    "_vmec_edge_r00_from_state",
    "_vmec_psia_from_indata",
    "_vmec_psia_from_state",
    "_vmec_volume_profiles_from_state",
]


def build_differentiable_neopax_field(
    *,
    n_r: int,
    rho_half,
    rho_full,
    volume_p,
    vp,
    iotaf,
    Psia,
    bmnc_b,
    rmnc_b,
    gmnc_b,
    xm_b,
    xn_b,
    bvco,
    buco,
    r0_override=None,
) -> DifferentiableNeopaxField:
    """Mirror `NEOPAX.Field(...)` with pure JAX array operations.

    The implementation intentionally follows the external constructor closely so
    raw-array parity can be checked directly, while avoiding NumPy scalar
    conversions that break JAX tracing.
    """

    n_r_int = int(n_r)
    rho_half_arr = jnp.asarray(rho_half)
    rho_full_arr = jnp.asarray(rho_full)
    vp_arr = jnp.asarray(vp)
    iotaf_arr = jnp.asarray(iotaf)
    bmnc_arr = jnp.asarray(bmnc_b)
    rmnc_arr = jnp.asarray(rmnc_b)
    gmnc_arr = jnp.asarray(gmnc_b)
    xm_arr = jnp.asarray(xm_b, dtype=jnp.int32)
    xn_arr = jnp.asarray(xn_b, dtype=jnp.int32)
    bvco_arr = jnp.asarray(bvco)
    buco_arr = jnp.asarray(buco)

    r0_value = jnp.asarray(r0_override) if r0_override is not None else rmnc_arr[-1, 0]
    a_b = jnp.sqrt(jnp.asarray(volume_p) / (2.0 * jnp.pi**2 * r0_value))

    rho_grid = jnp.linspace(0.0, 1.0, n_r_int)
    rho_grid_half0 = (
        0.5 * (rho_grid[0] + rho_grid[1])
        if n_r_int > 1
        else jnp.asarray(0.0, dtype=rho_grid.dtype)
    )
    rho_grid_half = jnp.linspace(rho_grid_half0, rho_grid_half0 + rho_grid[-1], n_r_int)
    r_grid = rho_grid * a_b
    r_grid_half = rho_grid_half * a_b
    dr = r_grid[2] - r_grid[1] if n_r_int > 2 else jnp.asarray(0.0, dtype=r_grid.dtype)

    idx00 = _find_mode_index(xm_arr, xn_arr, m_value=0, n_value=0)
    idx10 = _find_mode_index(xm_arr, xn_arr, m_value=1, n_value=0)

    b00 = interpax.Interpolator1D(rho_half_arr[1:], bmnc_arr[:, idx00], extrap=True)
    r00 = interpax.Interpolator1D(rho_full_arr[1:], rmnc_arr[:, idx00], extrap=True)
    sqrtg00 = interpax.Interpolator1D(rho_half_arr[1:], gmnc_arr[:, idx00], extrap=True)

    if idx10 is None:

        def b10_eval(x):
            return jnp.zeros_like(jnp.asarray(x))

    else:
        b10 = interpax.Interpolator1D(rho_half_arr[1:], bmnc_arr[:, idx10], extrap=True)

        def b10_eval(x):
            return b10(x)

    dVdr = interpax.Interpolator1D(rho_half_arr[1:], vp_arr[1:], extrap=True)
    vprime = dVdr(rho_grid) * 2.0 * rho_grid / a_b
    vprime_half = dVdr(rho_grid_half) * 2.0 * rho_grid_half / a_b
    over_vprime = _safe_reciprocal(vprime)
    over_vprime = over_vprime.at[0].set(0.0)

    iota_interp = interpax.Interpolator1D(rho_full_arr, iotaf_arr, extrap=True)
    iota = iota_interp(rho_grid)
    epsilon_t = rho_grid * a_b / r00(rho_grid)

    b00_rho = b00(rho_grid)
    b10_rho = b10_eval(rho_grid)
    b_10 = _safe_divide(b10_rho, b00_rho)
    b0 = b00(r_grid)
    b0prime = jax.lax.stop_gradient(jax.vmap(jax.grad(lambda r: b00(r)), in_axes=0)(r_grid))

    curvature = _safe_divide(jnp.abs(b_10), epsilon_t)
    curvature = jax.lax.stop_gradient(curvature.at[0].set(0.0))
    enlogation = jnp.square(_safe_divide(epsilon_t, b_10))
    enlogation = jax.lax.stop_gradient(enlogation.at[0].set(0.0))

    g_interp = interpax.Interpolator1D(rho_half_arr[1:], bvco_arr[1:], extrap=True)
    i_interp = interpax.Interpolator1D(rho_half_arr[1:], buco_arr[1:], extrap=True)
    g_value = g_interp(rho_grid)
    i_value = i_interp(rho_grid)

    d0 = 4.0 / 3.0
    d1 = 3.4229
    d2 = -2.5766
    d3 = -0.6039
    # The PS fit is orientation invariant: imported right-handed Boozer
    # conventions can flip the sign of iota without changing the geometry.
    iota_abs = jnp.abs(iota)
    iota_safe = jnp.where(iota_abs > 0.0, iota_abs, 1.0)
    g_ps = 1.5 * (
        d0
        * jnp.square(curvature / iota_safe)
        * (
            1.0
            + d1 * jnp.power(epsilon_t, 3.6) * (1.0 + d2 * jnp.power(iota_abs, 1.6))
            + d3 * jnp.power(epsilon_t, 2.0) * (1.0 - jnp.power(curvature, 2.0))
        )
    )
    g_ps = jax.lax.stop_gradient(g_ps)

    sqrtg00_value = sqrtg00(rho_grid_half)
    bsqav = _safe_divide(g_value + iota * i_value, sqrtg00_value * jnp.power(b0, 2.0))

    return DifferentiableNeopaxField(
        n_r=n_r_int,
        a_b=a_b,
        Psia_value=jnp.asarray(Psia),
        rho_grid=rho_grid,
        rho_grid_half=rho_grid_half,
        r_grid=r_grid,
        r_grid_half=r_grid_half,
        dr=dr,
        Vprime=vprime,
        Vprime_half=vprime_half,
        overVprime=over_vprime,
        epsilon_t=epsilon_t,
        B0=b0,
        B_10=b_10,
        enlogation=enlogation,
        iota=iota,
        R0=r0_value,
        B0prime=b0prime,
        curvature=curvature,
        G_PS=g_ps,
        sqrtg00_value=sqrtg00_value,
        Bsqav=bsqav,
        I_value=i_value,
        G_value=g_value,
    )
