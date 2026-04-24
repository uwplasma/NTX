"""Boozer-surface builders for optional ``vmec_jax`` workflows."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp

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


def surface_from_vmec_jax_wout(
    *,
    input_path: str | Path,
    wout_path: str | Path,
    s: float,
    mboz: int = 12,
    nboz: int = 12,
    psi_p: float = 1.0,
    min_bmn_to_load: float = 0.0,
) -> BoozerSurface:
    """Build a Boozer surface from a VMEC input file and matching `wout`.

    This helper keeps the workflow inside `vmec_jax` and `booz_xform_jax` while
    handling the common case in which the reference `wout` carries a higher
    radial resolution than the original VMEC input file. In that case the VMEC
    static configuration is rebuilt with `ns = wout.ns` so
    `booz_xform_inputs_from_state(...)` sees a consistent radial mesh.
    """

    import vmec_jax
    from vmec_jax.api import read_wout, state_from_wout

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
    return surface_from_vmec_jax_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(wout.signgs),
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
