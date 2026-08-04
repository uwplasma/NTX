"""The NEOPAX coupling: field containers, IO, flux exchange, and VMEX sourcing.

How NTX hands geometry and profiles to NEOPAX and reads fluxes back, including
building those fields directly from a VMEX equilibrium. The VMEX-sourced
builders and the field container they populate are mutually recursive, so they
share a module. Scans live in _neopax_scan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, tree_util

from ._interp import Interpolator1D
from ._vmex import (
    _apply_boozer_sign_convention_profiles,
    _booz_xform_bundle_from_vmex_state,
    _booz_xform_gmnc_from_inputs,
)
from .geometry import BoozerSurface, VmecSurface, evaluate_boozer_modes
from .vmex_backend import (
    VmecJaxBoundaryContext,
    solve_vmex_boundary_state,
    surfaces_from_vmex_state,
)

__all__ = [
    "_apply_boozer_sign_convention_profiles",
    "_booz_xform_bundle_with_gmnc_from_vmex_state",
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
    "_vmec_s_full",
    "_vmec_volume_profiles_from_state",
    "build_differentiable_neopax_field",
    "build_differentiable_neopax_field_from_vmec_booz_files",
    "build_differentiable_neopax_field_from_vmex_boundary_params",
    "build_differentiable_neopax_field_from_vmex_state",
]


# --- _neopax_vmex: NEOPAX fields sourced from a VMEX equilibrium. ---


def _booz_xform_bundle_with_gmnc_from_vmex_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    mboz: int,
    nboz: int,
):
    """Boozer bundle from a VMEX state, with the Jacobian harmonics included.

    gmnc is needed for flux-surface averages; it is a separate entry point
    because computing it costs an extra transform that most callers do not want.
    """
    inputs, out = _booz_xform_bundle_from_vmex_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        s_values=None,
        mboz=mboz,
        nboz=nboz,
    )
    gmnc_b = out.get("gmnc_b")
    if gmnc_b is None:
        gmnc_b = _booz_xform_gmnc_from_inputs(
            inputs=inputs,
            mboz=mboz,
            nboz=nboz,
            asym=bool(
                static.resolution.lasym if hasattr(static, "resolution") else static.cfg.lasym
            ),
        )
    out_with_gmnc = dict(out)
    out_with_gmnc["gmnc_b"] = gmnc_b
    return inputs, out_with_gmnc


def _vmec_s_full(static):
    """Full-mesh normalized toroidal flux, from whichever VMEX layout is present."""
    setup = getattr(static, "setup", None)
    return jnp.asarray(setup.s_full if setup is not None else static.s)


def _rho_half_mesh_from_s(s_full):
    """Half-mesh rho from the full-mesh s grid.

    VMEC stores different quantities on the full and half meshes; interpolating
    to the half mesh here keeps the radial coordinate consistent with the
    profiles that live there. Degenerate single-point grids pass straight
    through rather than producing an empty interior.
    """
    s_arr = jnp.asarray(s_full)
    if s_arr.shape[0] < 2:
        return jnp.sqrt(jnp.maximum(s_arr, 0.0))
    interior = 0.5 * (s_arr[1:] + s_arr[:-1])
    return jnp.concatenate(
        [
            jnp.zeros((1,), dtype=s_arr.dtype),
            jnp.sqrt(jnp.maximum(interior, 0.0)),
        ],
        axis=0,
    )


def _vmec_psia_from_state(state, static):
    """Toroidal flux at the edge, derived from the state's flux profile."""
    setup = getattr(static, "setup", None)
    if setup is not None:
        s_full = jnp.asarray(setup.s_full)
        phipf = jnp.asarray(setup.phipf)
        return jnp.abs(jnp.sum(phipf[1:] * jnp.diff(s_full)))

    phipf_out = getattr(state, "phipf_out", None)
    if phipf_out is None:
        raise AttributeError("vmex state does not expose `phipf_out`")

    try:
        from vmex.integrals import cumrect_s_halfmesh
    except ModuleNotFoundError as exc:  # pragma: no cover - local checkout fallback
        raise ImportError("vmex is required for differentiable field builders") from exc

    phi = cumrect_s_halfmesh(jnp.asarray(phipf_out), jnp.asarray(static.s))
    return jnp.abs(phi[-1])


def _vmec_edge_r00_from_state(state):
    """Major radius from the (0,0) harmonic at the boundary.

    Raises rather than guessing if the state exposes neither spelling of the
    cosine array: a silently wrong major radius rescales every transport result.
    """
    rcos = getattr(state, "R_cos", getattr(state, "Rcos", None))
    if rcos is None:
        raise AttributeError("vmex state does not expose `R_cos` or `Rcos`")
    return jnp.asarray(rcos)[-1, 0]


def _vmec_psia_from_indata(*, indata, static, signgs: int):
    """Edge toroidal flux from the input namelist, preferring phiedge when present."""
    if hasattr(indata, "phiedge"):
        return jnp.abs(jnp.asarray(indata.phiedge) / (2.0 * jnp.pi))

    try:
        from vmex.energy import flux_profiles_from_indata
        from vmex.integrals import cumrect_s_halfmesh
    except ModuleNotFoundError as exc:  # pragma: no cover - local checkout fallback
        raise ImportError("vmex is required for differentiable field builders") from exc

    flux = flux_profiles_from_indata(indata, jnp.asarray(static.s), signgs=int(signgs))
    phipf_out = jnp.asarray(flux.phipf)
    phi = cumrect_s_halfmesh(jnp.asarray(phipf_out), jnp.asarray(static.s))
    return jnp.abs(phi[-1])


def _vmec_volume_profiles_from_state(*, state, static, indata, signgs: int):
    """Volume and its derivative, recomputed from the VMEX state when possible.

    Imports from vmex lazily: the import is only reachable on the branch that
    needs it, so NTX stays usable without vmex installed.
    """
    if hasattr(static, "setup") and hasattr(state, "R_cos"):
        from vmex.core.fields import (
            energies_and_force_norms,
            magnetic_fields,
            metric_elements,
        )
        from vmex.core.geometry import half_mesh_jacobian
        from vmex.core.solver import _geometry

        setup = static.setup
        _, geometry = _geometry(state, static)
        jacobian = half_mesh_jacobian(geometry, s=setup.s_full)
        metrics = metric_elements(geometry, s=setup.s_full)
        fields = magnetic_fields(
            geometry=geometry,
            jacobian=jacobian,
            metrics=metrics,
            trig=static.trig,
            s=setup.s_full,
            phips=setup.phips,
            phipf=setup.phipf,
            chips=setup.chips,
            signgs=setup.signgs,
            gamma=static.gamma,
            mass=setup.mass,
            ncurr=setup.ncurr,
            enclosed_current=setup.icurv,
        )
        norms = energies_and_force_norms(
            jacobian=jacobian,
            metrics=metrics,
            fields=fields,
            trig=static.trig,
            s=setup.s_full,
            signgs=setup.signgs,
        )
        return (
            jnp.abs(jnp.asarray(norms.volume)) * (4.0 * jnp.pi**2),
            jnp.abs(jnp.asarray(norms.vp)),
        )

    from types import SimpleNamespace

    try:
        from vmex.vmec_forces import vmec_forces_rz_from_wout
        from vmex.vmec_residue import vmec_force_norms_from_bcovar_dynamic
    except ModuleNotFoundError as exc:  # pragma: no cover - local checkout fallback
        raise ImportError("vmex is required for differentiable field builders") from exc

    wout_like = SimpleNamespace(
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        signgs=int(signgs),
    )
    kernels = vmec_forces_rz_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        indata=indata,
        use_vmec_synthesis=True,
        trig=static.trig_vmec,
    )
    norms = vmec_force_norms_from_bcovar_dynamic(
        bc=kernels.bc,
        trig=static.trig_vmec,
        s=jnp.asarray(static.s),
        signgs=int(signgs),
    )
    return jnp.abs(jnp.asarray(norms.volume)) * (4.0 * jnp.pi**2), jnp.abs(jnp.asarray(norms.vp))


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


# --- _neopax: The NEOPAX coupling: field containers, IO, and flux exchange. ---

NEOPAX_SCAN_FORMAT_VERSION = 2


D33_MODES = frozenset({"spitzer", "raw", "conductivity_difference"})


@dataclass(frozen=True)
class NeopaxScan:
    """Monoenergetic scan data shaped for NEOPAX."""

    rho: Array
    nu_v: Array
    Er: Array
    Es: Array
    drds: Array
    D11: Array
    D13: Array
    D33: Array
    D33_spitzer: Array | None = None
    D31: Array | None = None
    Er_tilde: Array | None = None
    Er_to_Ertilde: Array | None = None
    dr_tildedr: Array | None = None
    dr_tildeds: Array | None = None
    a_b: float | None = None
    psia: float | None = None
    b00: Array | None = None
    r00: Array | None = None
    boozer_i: Array | None = None
    boozer_g: Array | None = None
    iota: Array | None = None
    fac_reference_to_sfincs_11: Array | None = None
    fac_reference_to_sfincs_31: Array | None = None
    fac_reference_to_sfincs_33: Array | None = None
    fac_monkes_to_sfincs_11: Array | None = None
    fac_monkes_to_sfincs_31: Array | None = None
    fac_monkes_to_sfincs_33: Array | None = None
    fac_sfincs_to_dkes_11: Array | None = None
    fac_sfincs_to_dkes_31: Array | None = None
    fac_sfincs_to_dkes_33: Array | None = None
    fac_dkes_to_d11star: Array | None = None
    fac_dkes_to_d31star: Array | None = None
    fac_dkes_to_d33star: Array | None = None
    source_name: str | None = None


tree_util.register_dataclass(
    NeopaxScan,
    data_fields=(
        "rho",
        "nu_v",
        "Er",
        "Es",
        "drds",
        "D11",
        "D13",
        "D33",
        "D33_spitzer",
        "D31",
        "Er_tilde",
        "Er_to_Ertilde",
        "dr_tildedr",
        "dr_tildeds",
        "a_b",
        "psia",
        "b00",
        "r00",
        "boozer_i",
        "boozer_g",
        "iota",
        "fac_reference_to_sfincs_11",
        "fac_reference_to_sfincs_31",
        "fac_reference_to_sfincs_33",
        "fac_monkes_to_sfincs_11",
        "fac_monkes_to_sfincs_31",
        "fac_monkes_to_sfincs_33",
        "fac_sfincs_to_dkes_11",
        "fac_sfincs_to_dkes_31",
        "fac_sfincs_to_dkes_33",
        "fac_dkes_to_d11star",
        "fac_dkes_to_d31star",
        "fac_dkes_to_d33star",
    ),
    meta_fields=("source_name",),
)


@dataclass(frozen=True)
class NeopaxMonoenergeticArrays:
    """Pure-array NEOPAX mapping payload for differentiable imported workflows."""

    a_b: Array
    rho: Array
    nu_log: Array
    Er_list: Array
    D11_log: Array
    D13: Array
    D33: Array


tree_util.register_dataclass(
    NeopaxMonoenergeticArrays,
    data_fields=("a_b", "rho", "nu_log", "Er_list", "D11_log", "D13", "D33"),
    meta_fields=(),
)


@dataclass(frozen=True)
class DifferentiableNeopaxField:
    """JAX-safe field payload compatible with the NEOPAX flux routines."""

    n_r: int
    a_b: Array
    Psia_value: Array
    rho_grid: Array
    rho_grid_half: Array
    r_grid: Array
    r_grid_half: Array
    dr: Array
    Vprime: Array
    Vprime_half: Array
    overVprime: Array
    epsilon_t: Array
    B0: Array
    B_10: Array
    enlogation: Array
    iota: Array
    R0: Array
    B0prime: Array
    curvature: Array
    G_PS: Array
    sqrtg00_value: Array
    Bsqav: Array
    I_value: Array
    G_value: Array


tree_util.register_dataclass(
    DifferentiableNeopaxField,
    data_fields=(
        "a_b",
        "Psia_value",
        "rho_grid",
        "rho_grid_half",
        "r_grid",
        "r_grid_half",
        "dr",
        "Vprime",
        "Vprime_half",
        "overVprime",
        "epsilon_t",
        "B0",
        "B_10",
        "enlogation",
        "iota",
        "R0",
        "B0prime",
        "curvature",
        "G_PS",
        "sqrtg00_value",
        "Bsqav",
        "I_value",
        "G_value",
    ),
    meta_fields=("n_r",),
)


def scan_to_neopax_arrays(
    scan: NeopaxScan,
    *,
    a_b: float | Array,
    d33_mode: str = "raw",
) -> NeopaxMonoenergeticArrays:
    """Map NTX scan data into the pure arrays consumed by `NEOPAX.Monoenergetic`.

    The default `raw` branch preserves the historical database convention used
    by the integrated workflow. The `spitzer` and `conductivity_difference`
    branches are explicit audit/stress-test choices and should not be promoted
    as global defaults without a transfer gate.
    """

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
            jnp.asarray(scan.D33_spitzer) if scan.D33_spitzer is not None else jnp.asarray(scan.D33)
        )
    elif d33_mode == "raw":
        d33 = jnp.asarray(scan.D33)
    else:
        if scan.D33_spitzer is None:
            raise ValueError("d33_mode='conductivity_difference' requires D33_spitzer in the scan")
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
    a_b: float | Array,
    d33_mode: str = "raw",
):
    """Construct `NEOPAX.Monoenergetic` from NTX scan data."""

    try:
        import NEOPAX
    except ImportError as exc:  # pragma: no cover - exercised when NEOPAX exists locally
        raise ImportError("NEOPAX is required for `to_neopax_monoenergetic`") from exc

    arrays = scan_to_neopax_arrays(scan, a_b=a_b, d33_mode=d33_mode)

    return NEOPAX.Monoenergetic(
        a_b=arrays.a_b,
        rho=arrays.rho,
        nu_log=arrays.nu_log,
        Er_list=arrays.Er_list,
        D11_log=arrays.D11_log,
        D13=arrays.D13,
        D33=arrays.D33,
    )


def _surface_transport_scale(surface: BoozerSurface | VmecSurface) -> Array:
    """Flux scale used to normalize transport coefficients, per surface type.

    VMEC and Boozer surfaces store this under different names for the same
    physical quantity.
    """
    if isinstance(surface, VmecSurface):
        return jnp.asarray(surface.transport_psi_scale, dtype=jnp.float64)
    return jnp.asarray(surface.psi_p, dtype=jnp.float64)


def _surface_reference_bridge(surface: BoozerSurface | VmecSurface) -> dict[str, Array]:
    """Reference field quantities bridging a surface into NEOPAX's conventions."""
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


def _safe_divide(num, den):
    """Divide, returning zero where the denominator is zero.

    Written with `where` on both the numerator and the denominator so the
    unused branch never evaluates a division by zero: under reverse-mode AD a
    NaN produced in a masked branch still propagates into the gradient.
    """
    num_arr = jnp.asarray(num)
    den_arr = jnp.asarray(den)
    den_safe = jnp.where(jnp.abs(den_arr) > 0.0, den_arr, 1.0)
    return jnp.where(jnp.abs(den_arr) > 0.0, num_arr / den_safe, 0.0)


def _safe_reciprocal(values):
    """Reciprocal, returning zero where the input is zero.

    Same masking discipline as `_safe_divide`, for the same AD reason.
    """
    arr = jnp.asarray(values)
    return jnp.where(jnp.abs(arr) > 0.0, 1.0 / arr, 0.0)


def _surface_b10(surface):
    """The B(1,0) harmonic amplitude, or zero when the surface lacks that mode."""
    mask = (jnp.asarray(surface.m) == 1) & (jnp.asarray(surface.n) == 0)
    idx = jnp.argmax(mask.astype(jnp.int32))
    b10 = jnp.where(mask.any(), jnp.asarray(surface.b_cos)[idx], 0.0)
    b0 = jnp.asarray(surface.b0 if surface.b0 is not None else surface.b_cos[0])
    return _safe_divide(b10, b0)


def _surface_bsqav(surface, *, ntheta: int = 31, nzeta: int = 31):
    """Flux-surface average of B squared, by direct quadrature on a theta-zeta grid.

    Only one field period is sampled in zeta, since the average over one period
    equals the average over all of them.
    """
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / int(surface.nfp), int(nzeta), endpoint=False)
    theta_2d, zeta_2d = jnp.meshgrid(theta, zeta, indexing="ij")
    b, _, _ = evaluate_boozer_modes(surface, theta_2d, zeta_2d)
    b0 = jnp.asarray(surface.b0 if surface.b0 is not None else surface.b_cos[0], dtype=b.dtype)
    inv_bsq_mean = jnp.mean(jnp.square(_safe_divide(b0, b)))
    return _safe_reciprocal(inv_bsq_mean)


def _find_mode_index(xm_b, xn_b, *, m_value: int, n_value: int) -> int | None:
    """Index of the (m, n) harmonic, or None when it is absent.

    Returns None rather than raising so callers can fall back to a default;
    the emptiness check is done in Python because the result picks a shape.
    """
    matches = (jnp.asarray(xm_b) == int(m_value)) & (jnp.asarray(xn_b) == int(n_value))
    if not bool(jnp.any(matches)):
        return None
    return int(jnp.argmax(matches))


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
        0.5 * (rho_grid[0] + rho_grid[1]) if n_r_int > 1 else jnp.asarray(0.0, dtype=rho_grid.dtype)
    )
    rho_grid_half = jnp.linspace(rho_grid_half0, rho_grid_half0 + rho_grid[-1], n_r_int)
    r_grid = rho_grid * a_b
    r_grid_half = rho_grid_half * a_b
    dr = r_grid[2] - r_grid[1] if n_r_int > 2 else jnp.asarray(0.0, dtype=r_grid.dtype)

    idx00 = _find_mode_index(xm_arr, xn_arr, m_value=0, n_value=0)
    idx10 = _find_mode_index(xm_arr, xn_arr, m_value=1, n_value=0)

    b00 = Interpolator1D(rho_half_arr[1:], bmnc_arr[:, idx00], method="akima")
    r00 = Interpolator1D(rho_full_arr[1:], rmnc_arr[:, idx00], method="akima")
    sqrtg00 = Interpolator1D(rho_half_arr[1:], gmnc_arr[:, idx00], method="akima")

    if idx10 is None:

        def b10_eval(x):
            return jnp.zeros_like(jnp.asarray(x))

    else:
        b10 = Interpolator1D(rho_half_arr[1:], bmnc_arr[:, idx10], method="akima")

        def b10_eval(x):
            return b10(x)

    dVdr = Interpolator1D(rho_half_arr[1:], vp_arr[1:], method="akima")
    vprime = dVdr(rho_grid) * 2.0 * rho_grid / a_b
    vprime_half = dVdr(rho_grid_half) * 2.0 * rho_grid_half / a_b
    over_vprime = _safe_reciprocal(vprime)
    over_vprime = over_vprime.at[0].set(0.0)

    iota_interp = Interpolator1D(rho_full_arr, iotaf_arr, method="akima")
    iota = iota_interp(rho_grid)
    epsilon_t = rho_grid * a_b / r00(rho_grid)

    b00_rho = b00(rho_grid)
    b10_rho = b10_eval(rho_grid)
    b_10 = _safe_divide(b10_rho, b00_rho)
    b0 = b00_rho
    d_b0_d_rho = jax.vmap(jax.grad(lambda rho_value: b00(rho_value)), in_axes=0)(rho_grid)
    b0prime = jax.lax.stop_gradient(_safe_divide(d_b0_d_rho, a_b))

    curvature = _safe_divide(jnp.abs(b_10), epsilon_t)
    curvature = jax.lax.stop_gradient(curvature.at[0].set(0.0))
    enlogation = jnp.square(_safe_divide(epsilon_t, b_10))
    enlogation = jax.lax.stop_gradient(enlogation.at[0].set(0.0))

    g_interp = Interpolator1D(rho_half_arr[1:], bvco_arr[1:], method="akima")
    i_interp = Interpolator1D(rho_half_arr[1:], buco_arr[1:], method="akima")
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


def _filled_dataset_array(handle, name: str):
    """Read a NetCDF variable, filling masked entries.

    netCDF4 returns masked arrays for variables with a fill value; unfilled,
    those masks propagate into JAX as silent NaNs.
    """
    values = handle.variables[name][:]
    if hasattr(values, "filled"):
        values = values.filled()
    return values


def build_differentiable_neopax_field_from_vmec_booz_files(
    n_r: int,
    vmec_path,
    booz_path,
) -> DifferentiableNeopaxField:
    """Build the NTX Boozer field object directly from VMEC and Boozer files.

    The Boozer coefficients are tabulated in normalized radius.  This reader uses
    `build_differentiable_neopax_field`, whose `B0` and `Bsqav` normalization is
    evaluated on that same normalized-radius support.
    """

    from netCDF4 import Dataset

    with Dataset(vmec_path, mode="r") as vmec, Dataset(booz_path, mode="r") as booz:
        ns = int(jnp.asarray(_filled_dataset_array(vmec, "ns")).reshape(()))
        s_full = jnp.linspace(0.0, 1.0, ns)
        s_half = jnp.asarray([(index - 0.5) / (ns - 1) for index in range(ns)])
        phi = jnp.asarray(_filled_dataset_array(vmec, "phi"))
        gmnc_name = "gmn_b" if "gmn_b" in booz.variables else "gmnc_b"
        return build_differentiable_neopax_field(
            n_r=n_r,
            rho_half=jnp.sqrt(jnp.clip(s_half, 0.0, None)),
            rho_full=jnp.sqrt(s_full),
            volume_p=jnp.asarray(_filled_dataset_array(vmec, "volume_p")),
            vp=jnp.asarray(_filled_dataset_array(vmec, "vp")),
            iotaf=jnp.asarray(_filled_dataset_array(vmec, "iotaf")),
            Psia=jnp.abs(phi[-1]),
            bmnc_b=jnp.asarray(_filled_dataset_array(booz, "bmnc_b")),
            rmnc_b=jnp.asarray(_filled_dataset_array(booz, "rmnc_b")),
            gmnc_b=jnp.asarray(_filled_dataset_array(booz, gmnc_name)),
            xm_b=jnp.asarray(_filled_dataset_array(booz, "ixm_b"), dtype=jnp.int32),
            xn_b=jnp.asarray(_filled_dataset_array(booz, "ixn_b"), dtype=jnp.int32),
            bvco=jnp.asarray(_filled_dataset_array(booz, "bvco_b")),
            buco=jnp.asarray(_filled_dataset_array(booz, "buco_b")),
        )


def get_differentiable_neopax_fluxes(species, grid, field, database):
    """Evaluate NEOPAX no-momentum fluxes with an axis-safe radial block.

    The reference monoenergetic databases do not include the magnetic axis
    exactly. For integrated objectives such as total bootstrap current, the
    axis contribution is weighted by `Vprime[0] = 0`, so copying the first
    interior radial block into the axis block removes an AD-only singularity
    without changing the physical integral.
    """

    try:
        from NEOPAX._neoclassical import get_Lij_matrix
    except ImportError as exc:  # pragma: no cover - exercised with local NEOPAX
        raise ImportError("NEOPAX is required for differentiable flux evaluation") from exc

    def _fluxes_internal(species_internal, species_index, lij):
        a1 = species_internal.A1[species_index]
        a2 = species_internal.A2[species_index]
        a3 = species_internal.A3
        temperature = species_internal.temperature[species_index]
        density = species_internal.density[species_index]
        gamma = -density * (
            lij[species_index, :, 0, 0] * a1
            + lij[species_index, :, 0, 1] * a2
            + lij[species_index, :, 0, 2] * a3
        )
        heat = (
            -temperature
            * density
            * (
                lij[species_index, :, 1, 0] * a1
                + lij[species_index, :, 1, 1] * a2
                + lij[species_index, :, 1, 2] * a3
            )
        )
        upar = -density * (
            lij[species_index, :, 2, 0] * a1
            + lij[species_index, :, 2, 1] * a2
            + lij[species_index, :, 2, 2] * a3
        )
        return gamma, heat, upar

    radial_indices = jnp.asarray(grid.full_grid_indeces)
    interior_indices = radial_indices[1:]
    lij_interior = jax.vmap(
        jax.vmap(get_Lij_matrix, in_axes=(None, None, None, None, None, 0)),
        in_axes=(None, None, None, None, 0, None),
    )(species, grid, field, database, species.species_indeces, interior_indices)
    lij = jnp.concatenate([lij_interior[:, :1, :, :], lij_interior], axis=1)
    gamma, heat, upar = jax.vmap(_fluxes_internal, in_axes=(None, 0, None))(
        species,
        species.species_indeces,
        lij,
    )
    return lij, gamma, heat, upar


def load_neopax_reference_scan(path: str | Path) -> NeopaxScan:
    """Load a NEOPAX-style HDF5 monoenergetic table."""

    import h5py

    h5_path = Path(path).expanduser().resolve()
    with h5py.File(h5_path, "r") as handle:
        return NeopaxScan(
            rho=jnp.asarray(handle["rho"][()]),
            nu_v=jnp.asarray(handle["nu_v"][()]),
            Er=jnp.asarray(handle["Er"][()]),
            Es=jnp.asarray(handle["Es"][()]),
            drds=jnp.asarray(handle["drds"][()]),
            D11=jnp.asarray(handle["D11"][()]),
            D13=jnp.asarray(handle["D13"][()]),
            D33=jnp.asarray(handle["D33"][()]),
            D33_spitzer=_optional_dataset(handle, "D33_spitzer"),
            D31=_optional_dataset(handle, "D31"),
            Er_tilde=_optional_dataset(handle, "Er_tilde"),
            Er_to_Ertilde=_optional_dataset(handle, "Er_to_Ertilde"),
            dr_tildedr=_optional_dataset(handle, "dr_tildedr"),
            dr_tildeds=_optional_dataset(handle, "dr_tildeds"),
            b00=_optional_dataset(handle, "B00"),
            r00=_optional_dataset(handle, "R00"),
            boozer_i=_optional_dataset(handle, "I"),
            boozer_g=_optional_dataset(handle, "G"),
            iota=_optional_dataset(handle, "iota"),
            fac_reference_to_sfincs_11=_optional_dataset(handle, "Fac_REFERENCE_TO_SFINCS_11"),
            fac_reference_to_sfincs_31=_optional_dataset(handle, "Fac_REFERENCE_TO_SFINCS_31"),
            fac_reference_to_sfincs_33=_optional_dataset(handle, "Fac_REFERENCE_TO_SFINCS_33"),
            fac_monkes_to_sfincs_11=_optional_dataset(handle, "Fac_MONKES_TO_SFINCS_11"),
            fac_monkes_to_sfincs_31=_optional_dataset(handle, "Fac_MONKES_TO_SFINCS_31"),
            fac_monkes_to_sfincs_33=_optional_dataset(handle, "Fac_MONKES_TO_SFINCS_33"),
            fac_sfincs_to_dkes_11=_optional_dataset(handle, "Fac_SFINCS_TO_DKES_11"),
            fac_sfincs_to_dkes_31=_optional_dataset(handle, "Fac_SFINCS_TO_DKES_31"),
            fac_sfincs_to_dkes_33=_optional_dataset(handle, "Fac_SFINCS_TO_DKES_33"),
            fac_dkes_to_d11star=_optional_dataset(handle, "Fac_DKES_TO_D11star"),
            fac_dkes_to_d31star=_optional_dataset(handle, "Fac_DKES_TO_D31star"),
            fac_dkes_to_d33star=_optional_dataset(handle, "Fac_DKES_TO_D33star"),
            source_name=str(handle.attrs.get("source_name", h5_path.name)),
        )


def neopax_scan_requires_rebuild(path: str | Path) -> bool:
    """Return whether a cached NEOPAX-style scan is missing required fields."""

    import h5py

    h5_path = Path(path).expanduser().resolve()
    if not h5_path.exists():
        return True
    with h5py.File(h5_path, "r") as handle:
        format_version = int(handle.attrs.get("format_version", 0))
        return format_version < NEOPAX_SCAN_FORMAT_VERSION or "D33_spitzer" not in handle


def write_neopax_scan_hdf5(scan: NeopaxScan, path: str | Path) -> Path:
    """Write a NEOPAX-style HDF5 file from a scan payload."""

    import h5py

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        for name, values in _scan_datasets(scan):
            _write_dataset(handle, name, values)
        if scan.a_b is not None:
            handle.attrs["a_b"] = float(scan.a_b)
        if scan.psia is not None:
            handle.attrs["psia"] = float(scan.psia)
        if scan.source_name is not None:
            handle.attrs["source_name"] = scan.source_name
        handle.attrs["format_version"] = NEOPAX_SCAN_FORMAT_VERSION
    return output_path


def _optional_dataset(handle, name: str):
    """Read an HDF5 dataset if present, otherwise None."""
    if name not in handle:
        return None
    return jnp.asarray(handle[name][()])


def _write_dataset(handle, name: str, values) -> None:
    """Write a dataset, skipping None.

    `track_times=False` keeps the file byte-identical across runs, so an output
    can be checksummed and compared.
    """
    if values is None:
        return
    handle.create_dataset(name, data=np.asarray(values), track_times=False)


def _scan_datasets(scan: NeopaxScan):
    """The (name, array) pairs that make up a scan file.

    One list drives both writing and reading, so the two cannot drift apart.
    """
    return (
        ("rho", scan.rho),
        ("nu_v", scan.nu_v),
        ("Er", scan.Er),
        ("Es", scan.Es),
        ("drds", scan.drds),
        ("D11", scan.D11),
        ("D13", scan.D13),
        ("D33", scan.D33),
        ("D33_spitzer", scan.D33_spitzer),
        ("D31", scan.D31),
        ("Er_tilde", scan.Er_tilde),
        ("Er_to_Ertilde", scan.Er_to_Ertilde),
        ("dr_tildedr", scan.dr_tildedr),
        ("dr_tildeds", scan.dr_tildeds),
        ("B00", scan.b00),
        ("R00", scan.r00),
        ("I", scan.boozer_i),
        ("G", scan.boozer_g),
        ("iota", scan.iota),
        ("Fac_REFERENCE_TO_SFINCS_11", scan.fac_reference_to_sfincs_11),
        ("Fac_REFERENCE_TO_SFINCS_31", scan.fac_reference_to_sfincs_31),
        ("Fac_REFERENCE_TO_SFINCS_33", scan.fac_reference_to_sfincs_33),
        ("Fac_MONKES_TO_SFINCS_11", scan.fac_monkes_to_sfincs_11),
        ("Fac_MONKES_TO_SFINCS_31", scan.fac_monkes_to_sfincs_31),
        ("Fac_MONKES_TO_SFINCS_33", scan.fac_monkes_to_sfincs_33),
        ("Fac_SFINCS_TO_DKES_11", scan.fac_sfincs_to_dkes_11),
        ("Fac_SFINCS_TO_DKES_31", scan.fac_sfincs_to_dkes_31),
        ("Fac_SFINCS_TO_DKES_33", scan.fac_sfincs_to_dkes_33),
        ("Fac_DKES_TO_D11star", scan.fac_dkes_to_d11star),
        ("Fac_DKES_TO_D31star", scan.fac_dkes_to_d31star),
        ("Fac_DKES_TO_D33star", scan.fac_dkes_to_d33star),
    )
