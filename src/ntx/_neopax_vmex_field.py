"""Differentiable NEOPAX field builders backed by vmex states."""

from __future__ import annotations

from ._interp import Interpolator1D

import jax
import jax.numpy as jnp

from ._neopax_field_utils import (
    _safe_divide,
    _safe_reciprocal,
    _surface_b10,
    _surface_bsqav,
)
from ._neopax_types import DifferentiableNeopaxField
from ._neopax_vmex_boozer import (
    _booz_xform_bundle_with_gmnc_from_vmex_state,
    _booz_xform_gmnc_from_inputs,
)
from ._neopax_vmex_profiles import (
    _rho_half_mesh_from_s,
    _vmec_edge_r00_from_state,
    _vmec_psia_from_indata,
    _vmec_psia_from_state,
    _vmec_s_full,
    _vmec_volume_profiles_from_state,
)
from ._vmex_boozer import (
    _apply_boozer_sign_convention_profiles,
)
from .vmex_backend import (
    VmecJaxBoundaryContext,
    solve_vmex_boundary_state,
    surfaces_from_vmex_state,
)

__all__ = [
    "build_differentiable_neopax_field_from_vmex_boundary_params",
    "build_differentiable_neopax_field_from_vmex_state",
    "_apply_boozer_sign_convention_profiles",
    "_booz_xform_bundle_with_gmnc_from_vmex_state",
    "_booz_xform_gmnc_from_inputs",
    "_rho_half_mesh_from_s",
    "_vmec_edge_r00_from_state",
    "_vmec_psia_from_indata",
    "_vmec_psia_from_state",
    "_vmec_volume_profiles_from_state",
]


def build_differentiable_neopax_field_from_vmex_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    n_r: int,
    mboz: int = 12,
    nboz: int = 12,
    apply_boozer_sign_convention: bool = True,
) -> DifferentiableNeopaxField:
    """Build a tracer-safe NEOPAX field from an in-memory `vmex` state."""

    s_full = _vmec_s_full(static)
    rho_half = _rho_half_mesh_from_s(s_full)

    volume_p, vp = _vmec_volume_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
    )
    phipf_out = getattr(state, "phipf_out", None)
    psia = (
        _vmec_psia_from_state(state, static)
        if phipf_out is not None
        else _vmec_psia_from_indata(indata=indata, static=static, signgs=signgs)
    )
    n_r_int = int(n_r)
    r0_value = _vmec_edge_r00_from_state(state)
    a_b = jnp.sqrt(jnp.asarray(volume_p) / (2.0 * jnp.pi**2 * r0_value))

    rho_grid = jnp.linspace(0.0, 1.0, n_r_int)
    rho_grid_half0 = (
        0.5 * (rho_grid[0] + rho_grid[1]) if n_r_int > 1 else jnp.asarray(0.0, dtype=rho_grid.dtype)
    )
    rho_grid_half = jnp.linspace(rho_grid_half0, rho_grid_half0 + rho_grid[-1], n_r_int)
    r_grid = rho_grid * a_b
    if n_r_int > 1:
        r_grid = r_grid.at[0].set(0.5 * r_grid[1])
    r_grid_half = rho_grid_half * a_b
    dr = r_grid[2] - r_grid[1] if n_r_int > 2 else jnp.asarray(0.0, dtype=r_grid.dtype)

    dVdr = Interpolator1D(rho_half[1:], jnp.asarray(vp)[1:], method="akima")
    vprime = dVdr(rho_grid) * 2.0 * rho_grid / a_b
    vprime_half = dVdr(rho_grid_half) * 2.0 * rho_grid_half / a_b
    over_vprime = _safe_reciprocal(vprime)
    over_vprime = over_vprime.at[0].set(0.0)

    sample_rho = rho_grid[1:-1]
    sample_surfaces = surfaces_from_vmex_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        s_values=tuple(float(rho_value**2) for rho_value in sample_rho),
        mboz=mboz,
        nboz=nboz,
        psi_p=float(psia),
    )

    iota_samples = jnp.asarray([jnp.asarray(surface.iota) for surface in sample_surfaces])
    i_value_samples = jnp.asarray([jnp.asarray(surface.b_theta) for surface in sample_surfaces])
    g_value_samples = jnp.asarray([jnp.asarray(surface.b_zeta) for surface in sample_surfaces])
    b0_samples = jnp.asarray(
        [
            jnp.asarray(surface.b0 if surface.b0 is not None else surface.b_cos[0])
            for surface in sample_surfaces
        ]
    )
    b10_samples = jnp.asarray([_surface_b10(surface) for surface in sample_surfaces])
    bsqav_samples = jnp.asarray([_surface_bsqav(surface) for surface in sample_surfaces])

    iota = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=iota_samples.dtype),
            iota_samples,
            iota_samples[-1:],
        ],
        axis=0,
    )
    i_value = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=i_value_samples.dtype),
            i_value_samples,
            i_value_samples[-1:],
        ],
        axis=0,
    )
    g_value = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=g_value_samples.dtype),
            g_value_samples,
            g_value_samples[-1:],
        ],
        axis=0,
    )
    b0 = jax.lax.stop_gradient(
        jnp.concatenate([b0_samples[:1], b0_samples, b0_samples[-1:]], axis=0)
    )
    b_10 = jax.lax.stop_gradient(
        jnp.concatenate(
            [
                jnp.zeros((1,), dtype=b10_samples.dtype),
                b10_samples,
                b10_samples[-1:],
            ],
            axis=0,
        )
    )
    bsqav = jnp.concatenate(
        [
            jnp.ones((1,), dtype=bsqav_samples.dtype),
            bsqav_samples,
            bsqav_samples[-1:],
        ],
        axis=0,
    )

    epsilon_t = jax.lax.stop_gradient(rho_grid * a_b / r0_value)
    curvature = jax.lax.stop_gradient(_safe_divide(jnp.abs(b_10), epsilon_t).at[0].set(0.0))
    enlogation = jax.lax.stop_gradient(jnp.square(_safe_divide(epsilon_t, b_10)).at[0].set(0.0))
    b0prime = jnp.zeros_like(rho_grid)
    iota_abs = jnp.abs(iota)
    iota_safe = jnp.where(iota_abs > 0.0, iota_abs, 1.0)
    g_ps = jax.lax.stop_gradient(
        1.5
        * (4.0 / 3.0)
        * jnp.square(curvature / iota_safe)
        * (
            1.0
            + 3.4229 * jnp.power(epsilon_t, 3.6) * (1.0 - 2.5766 * jnp.power(iota_abs, 1.6))
            - 0.6039 * jnp.power(epsilon_t, 2.0) * (jnp.power(curvature, 2.0) - 1.0)
        )
    )

    sqrtg00_full = _safe_divide(g_value + iota * i_value, bsqav * jnp.square(b0))
    sqrtg00_interp = Interpolator1D(rho_grid, sqrtg00_full, method="akima")
    sqrtg00_value = jax.lax.stop_gradient(sqrtg00_interp(rho_grid_half))

    return DifferentiableNeopaxField(
        n_r=n_r_int,
        a_b=a_b,
        Psia_value=jnp.asarray(psia),
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


def build_differentiable_neopax_field_from_vmex_boundary_params(
    context: VmecJaxBoundaryContext,
    params,
    *,
    n_r: int,
    vmec_project: bool = True,
    max_iter: int = 50,
    step_size: float = 1.0,
    ftol: float | None = None,
    implicit=None,
    mboz: int = 12,
    nboz: int = 12,
    apply_boozer_sign_convention: bool = True,
) -> DifferentiableNeopaxField:
    """Solve a fixed boundary and build a tracer-safe NEOPAX field."""

    state = solve_vmex_boundary_state(
        context,
        params,
        vmec_project=vmec_project,
        max_iter=max_iter,
        step_size=step_size,
        ftol=ftol,
        implicit=implicit,
    )
    return build_differentiable_neopax_field_from_vmex_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=context.signgs,
        n_r=n_r,
        mboz=mboz,
        nboz=nboz,
        apply_boozer_sign_convention=apply_boozer_sign_convention,
    )
