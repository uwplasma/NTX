"""Boozer-surface builders for optional ``vmec_jax`` workflows."""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ._checkout_paths import find_booz_xform_jax_root
from ._vmec_jax_boozer import (
    _apply_boozer_sign_convention,
    _booz_xform_bundle_from_vmec_jax_state,
)
from ._vmec_jax_boundary import VmecJaxBoundaryContext, solve_vmec_jax_boundary_state
from .geometry import BoozerSurface


def surface_from_vmec_jax_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    s: float,
    mboz: int = 12,
    nboz: int = 12,
    psi_p: float = 1.0,
    min_bmn_to_load: float = 0.0,
    flux_profiles=None,
    profiles_half=None,
) -> BoozerSurface:
    """Build a Boozer surface from in-memory `vmec_jax` state.

    This is the differentiable imported lane: VMEC state stays in Python/JAX
    memory, the Boozer transform is done with `booz_xform_jax`, and NTX solves
    directly from the resulting Boozer harmonics.
    """
    return surfaces_from_vmec_jax_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        s_values=(s,),
        mboz=mboz,
        nboz=nboz,
        psi_p=psi_p,
        min_bmn_to_load=min_bmn_to_load,
        flux_profiles=flux_profiles,
        profiles_half=profiles_half,
    )[0]


def surfaces_from_vmec_jax_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    s_values: Sequence[float],
    mboz: int = 12,
    nboz: int = 12,
    psi_p: float = 1.0,
    min_bmn_to_load: float = 0.0,
    flux_profiles=None,
    profiles_half=None,
) -> tuple[BoozerSurface, ...]:
    """Build several Boozer surfaces from one in-memory `vmec_jax` state."""

    inputs, out = _booz_xform_bundle_from_vmec_jax_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        s_values=s_values,
        mboz=mboz,
        nboz=nboz,
        flux_profiles=flux_profiles,
        profiles_half=profiles_half,
    )
    ixm_b = jnp.asarray(out["ixm_b"], dtype=jnp.int32)
    ixn_b = jnp.asarray(out["ixn_b"], dtype=jnp.int32)
    bmnc_all = jnp.asarray(out["bmnc_b"])
    source_value = getattr(indata, "input_filename", "vmec_jax_state")
    source_path = Path(str(source_value)).expanduser()
    surfaces: list[BoozerSurface] = []
    for row in range(int(bmnc_all.shape[0])):
        bmnc_b = bmnc_all[row]
        iota, b_theta, b_zeta = _apply_boozer_sign_convention(
            iota=jnp.asarray(out["iota_b"])[row],
            b_theta=jnp.asarray(out["buco_b"])[row],
            b_zeta=jnp.asarray(out["bvco_b"])[row],
        )
        b0 = bmnc_b[0]
        include = jnp.abs(bmnc_b / b0) >= jnp.asarray(min_bmn_to_load, dtype=bmnc_b.dtype)
        include = include.at[0].set(True)
        surfaces.append(
            BoozerSurface(
                m=ixm_b[include],
                n=jnp.asarray(jnp.rint(ixn_b[include] / int(inputs.nfp)), dtype=jnp.int32),
                b_cos=bmnc_b[include],
                nfp=int(inputs.nfp),
                iota=iota,
                psi_p=jnp.asarray(psi_p, dtype=bmnc_b.dtype),
                b_theta=b_theta,
                b_zeta=b_zeta,
                b0=b0,
                source_path=source_path,
            )
        )
    return tuple(surfaces)


def surfaces_from_vmec_jax_boundary_params(
    context: VmecJaxBoundaryContext,
    params,
    *,
    s_values: Sequence[float],
    vmec_project: bool = True,
    max_iter: int = 50,
    step_size: float = 1.0,
    ftol: float | None = None,
    implicit=None,
    mboz: int = 12,
    nboz: int = 12,
    psi_p: float = 1.0,
    min_bmn_to_load: float = 0.0,
) -> tuple[BoozerSurface, ...]:
    """Solve a fixed boundary and return the requested Boozer surfaces."""

    state = solve_vmec_jax_boundary_state(
        context,
        params,
        vmec_project=vmec_project,
        max_iter=max_iter,
        step_size=step_size,
        ftol=ftol,
        implicit=implicit,
    )
    return surfaces_from_vmec_jax_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=context.signgs,
        s_values=s_values,
        mboz=mboz,
        nboz=nboz,
        psi_p=psi_p,
        min_bmn_to_load=min_bmn_to_load,
    )


def _wout_flux_and_half_profiles(wout, static, *, signgs: int):
    """Build Boozer flux/profile inputs from finalized VMEC `wout` channels."""

    from vmec_jax.energy import FluxProfiles, lamscale_from_phips

    required = ("phipf", "chipf", "phips", "iotas")
    missing = [name for name in required if getattr(wout, name, None) is None]
    if missing:
        raise ValueError(
            "wout is missing Boozer-profile channels needed for fallback: "
            + ", ".join(missing)
        )

    phips = jnp.asarray(wout.phips)
    s_full = jnp.asarray(static.s, dtype=phips.dtype)
    if phips.shape[0] != s_full.shape[0]:
        raise ValueError(
            "wout phips and VMEC static radial grids have inconsistent lengths: "
            f"{phips.shape[0]} != {s_full.shape[0]}"
        )
    iota = jnp.asarray(wout.iotas, dtype=phips.dtype)
    if iota.shape[0] != s_full.shape[0]:
        raise ValueError(
            "wout iotas and VMEC static radial grids have inconsistent lengths: "
            f"{iota.shape[0]} != {s_full.shape[0]}"
        )

    pressure_source = getattr(wout, "pres", None)
    pressure = (
        jnp.zeros_like(iota)
        if pressure_source is None
        else jnp.asarray(pressure_source, dtype=phips.dtype)
    )
    if pressure.shape[0] != s_full.shape[0]:
        raise ValueError(
            "wout pressure and VMEC static radial grids have inconsistent lengths: "
            f"{pressure.shape[0]} != {s_full.shape[0]}"
        )

    flux = FluxProfiles(
        phipf=int(signgs) * jnp.asarray(wout.phipf, dtype=phips.dtype) / (2.0 * jnp.pi),
        chipf=int(signgs) * jnp.asarray(wout.chipf, dtype=phips.dtype) / (2.0 * jnp.pi),
        phips=phips,
        signgs=int(signgs),
        lamscale=lamscale_from_phips(phips, s_full),
    )
    return flux, {"iota": iota, "pressure": pressure}


def _surface_from_booz_xform_wout_data(
    wout,
    *,
    source_path: Path,
    s: float,
    mboz: int,
    nboz: int,
    psi_p: float,
    min_bmn_to_load: float,
) -> BoozerSurface:
    """Build a Boozer surface from finalized VMEC `wout` magnetic channels."""

    try:
        from booz_xform_jax import Booz_xform
    except ModuleNotFoundError:
        checkout = find_booz_xform_jax_root()
        if checkout is not None and str(checkout) not in sys.path:
            sys.path.insert(0, str(checkout))
        from booz_xform_jax import Booz_xform

    bx = Booz_xform()
    bx.verbose = 0
    bx.read_wout_data(wout, flux=True)
    s_grid = np.asarray(bx.s_in, dtype=float).reshape(-1)
    if s_grid.size == 0:
        raise ValueError("wout-backed Boozer transform has no radial surfaces")
    idx = int(np.argmin(np.abs(s_grid - float(s))))
    bx.compute_surfs = [idx]
    bx.mboz = int(mboz)
    bx.nboz = int(nboz)
    out = bx.run_jax(jit=True)

    xm = np.asarray(out["ixm_b"], dtype=np.int32).reshape(-1)
    xn = np.asarray(out["ixn_b"], dtype=np.int32).reshape(-1)
    bmn = np.asarray(out["bmnc_b"], dtype=np.float64)[0]
    nfp = int(np.asarray(out["nfp_b"]).reshape(()))
    iota_value = -float(np.asarray(out["iota_b"], dtype=np.float64).reshape(-1)[0])
    buco_value = -float(np.asarray(out["buco_b"], dtype=np.float64).reshape(-1)[0])
    bvco_value = float(np.asarray(out["bvco_b"], dtype=np.float64).reshape(-1)[0])
    sign = 1.0 if (bvco_value + iota_value * buco_value) >= 0.0 else -1.0
    buco_value *= sign
    bvco_value *= sign

    b0 = float(bmn[0])
    if b0 == 0.0:
        raise ValueError("Boozer mode (m,n)=(0,0) is zero on the selected surface")
    include = np.abs(bmn / b0) >= float(min_bmn_to_load)
    include[0] = True

    return BoozerSurface(
        m=jnp.asarray(xm[include], dtype=jnp.int32),
        n=jnp.asarray(np.rint(xn[include] / nfp).astype(np.int32), dtype=jnp.int32),
        b_cos=jnp.asarray(bmn[include]),
        nfp=nfp,
        iota=iota_value,
        psi_p=jnp.asarray(psi_p, dtype=jnp.asarray(bmn).dtype),
        b_theta=buco_value,
        b_zeta=bvco_value,
        b0=b0,
        source_path=source_path,
    )


def surface_from_vmec_jax_wout(
    *,
    input_path: str | Path,
    wout_path: str | Path,
    s: float,
    mboz: int = 12,
    nboz: int = 12,
    psi_p: float = 1.0,
    min_bmn_to_load: float = 0.0,
    profile_source: str = "auto",
) -> BoozerSurface:
    """Build a Boozer surface from a VMEC input file and matching `wout`.

    This helper keeps the workflow inside `vmec_jax` and `booz_xform_jax` while
    handling the common case in which the reference `wout` carries a higher
    radial resolution than the original VMEC input file. In that case the VMEC
    static configuration is rebuilt with `ns = wout.ns` so
    `booz_xform_inputs_from_state(...)` sees a consistent radial mesh.

    `profile_source="auto"` first uses the differentiable VMEC-state path and
    falls back to finalized `wout` magnetic channels when the input uses a
    profile representation that the optional JAX VMEC stack cannot re-evaluate.
    `profile_source="wout"` selects that finalized magnetic-channel path
    explicitly. It is the correct file-backed path for unsupported optimized
    finite-beta inputs, but it is not a differentiable equilibrium-state path.
    `profile_source="state_wout_profiles"` is a diagnostic path that injects
    finalized `wout` flux and half-grid profiles into the state transform while
    still using the state-reconstructed magnetic channels.
    """

    import vmec_jax
    from vmec_jax.api import read_wout, state_from_wout

    if profile_source not in {"auto", "input", "wout", "state_wout_profiles"}:
        raise ValueError(
            "profile_source must be 'auto', 'input', 'wout', or 'state_wout_profiles'"
        )

    vmec_input = Path(input_path).expanduser().resolve()
    vmec_wout = Path(wout_path).expanduser().resolve()
    cfg, indata = vmec_jax.load_config(vmec_input)
    wout = read_wout(vmec_wout)
    state = state_from_wout(wout)

    replacements: dict[str, int] = {}
    if int(cfg.ns) != int(wout.ns):
        replacements["ns"] = int(wout.ns)
    if int(cfg.mpol) != int(wout.mpol):
        replacements["mpol"] = int(wout.mpol)
    if int(cfg.ntor) != int(wout.ntor):
        replacements["ntor"] = int(wout.ntor)
    if replacements:
        cfg = dataclasses.replace(cfg, **replacements)

    static = vmec_jax.build_static(cfg)
    signgs = int(wout.signgs)

    def build_surface(*, use_wout_profiles: bool):
        flux_profiles = None
        profiles_half = None
        if use_wout_profiles:
            flux_profiles, profiles_half = _wout_flux_and_half_profiles(
                wout,
                static,
                signgs=signgs,
            )
        return surface_from_vmec_jax_state(
            state=state,
            static=static,
            indata=indata,
            signgs=signgs,
            s=s,
            mboz=mboz,
            nboz=nboz,
            psi_p=psi_p,
            min_bmn_to_load=min_bmn_to_load,
            flux_profiles=flux_profiles,
            profiles_half=profiles_half,
        )

    if profile_source == "wout":
        return _surface_from_booz_xform_wout_data(
            wout,
            source_path=vmec_wout,
            s=s,
            mboz=mboz,
            nboz=nboz,
            psi_p=psi_p,
            min_bmn_to_load=min_bmn_to_load,
        )
    if profile_source == "state_wout_profiles":
        return build_surface(use_wout_profiles=True)
    try:
        return build_surface(use_wout_profiles=False)
    except NotImplementedError:
        if profile_source != "auto":
            raise
        return _surface_from_booz_xform_wout_data(
            wout,
            source_path=vmec_wout,
            s=s,
            mboz=mboz,
            nboz=nboz,
            psi_p=psi_p,
            min_bmn_to_load=min_bmn_to_load,
        )


__all__ = [
    "surface_from_vmec_jax_state",
    "surface_from_vmec_jax_wout",
    "surfaces_from_vmec_jax_boundary_params",
    "surfaces_from_vmec_jax_state",
]
