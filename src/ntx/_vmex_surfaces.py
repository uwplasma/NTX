"""Boozer-surface builders for optional ``vmex`` workflows."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ._checkout_paths import find_booz_xform_jax_root
from ._vmex_boozer import (
    _apply_boozer_sign_convention,
    _booz_xform_bundle_from_vmex_state,
)
from ._vmex_boundary import VmecJaxBoundaryContext, solve_vmex_boundary_state
from .geometry import BoozerSurface


def surface_from_vmex_state(
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
    """Build a Boozer surface from in-memory `vmex` state.

    This is the differentiable imported lane: VMEC state stays in Python/JAX
    memory, the Boozer transform is done with `booz_xform_jax`, and NTX solves
    directly from the resulting Boozer harmonics.
    """
    return surfaces_from_vmex_state(
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


def surfaces_from_vmex_state(
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
    """Build several Boozer surfaces from one in-memory `vmex` state."""

    inputs, out = _booz_xform_bundle_from_vmex_state(
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
    source_value = getattr(
        indata,
        "input_filename",
        getattr(indata, "source_path", "vmex_state"),
    )
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


def surfaces_from_vmex_boundary_params(
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

    state = solve_vmex_boundary_state(
        context,
        params,
        vmec_project=vmec_project,
        max_iter=max_iter,
        step_size=step_size,
        ftol=ftol,
        implicit=implicit,
    )
    return surfaces_from_vmex_state(
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


def surface_from_vmex_wout(
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

    ``profile_source="auto"`` and ``"wout"`` transform the finalized WOUT
    magnetic channels. This is the physically consistent file-backed path and
    does not reconstruct a VMEC state from output coefficients. Differentiable
    workflows should call :func:`surface_from_vmex_state` with the current
    vmex ``SpectralState`` and matching ``SolverRuntime``.
    """

    try:
        from vmex import read_wout
    except (ImportError, ModuleNotFoundError):
        try:
            from vmex.api import read_wout
        except (ImportError, ModuleNotFoundError) as exc:
            raise ModuleNotFoundError(
                "surface_from_vmex_wout requires vmex. "
                "Install it with `pip install vmex`."
            ) from exc

    if profile_source not in {"auto", "input", "wout", "state_wout_profiles"}:
        raise ValueError("profile_source must be 'auto', 'input', 'wout', or 'state_wout_profiles'")

    vmec_input = Path(input_path).expanduser().resolve()
    vmec_wout = Path(wout_path).expanduser().resolve()
    if not vmec_input.exists():
        raise FileNotFoundError(str(vmec_input))
    if not vmec_wout.exists():
        raise FileNotFoundError(str(vmec_wout))
    wout = read_wout(vmec_wout)
    if profile_source in {"auto", "wout"}:
        return _surface_from_booz_xform_wout_data(
            wout,
            source_path=vmec_wout,
            s=s,
            mboz=mboz,
            nboz=nboz,
            psi_p=psi_p,
            min_bmn_to_load=min_bmn_to_load,
        )
    raise NotImplementedError(
        f"profile_source={profile_source!r} depended on the removed legacy "
        "vmex state_from_wout API. Use profile_source='wout' for files, "
        "or surface_from_vmex_state(...) with a current vmex "
        "SpectralState and SolverRuntime for differentiable calculations."
    )


__all__ = [
    "surface_from_vmex_state",
    "surface_from_vmex_wout",
    "surfaces_from_vmex_boundary_params",
    "surfaces_from_vmex_state",
]
