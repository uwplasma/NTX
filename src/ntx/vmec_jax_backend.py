"""Helpers for `vmec_jax -> booz_xform_jax -> NTX` workflows."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax.numpy as jnp

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

    from booz_xform_jax.jax_api import (
        booz_xform_from_inputs,
        prepare_booz_xform_constants_from_inputs,
    )
    from vmec_jax import booz_xform_inputs_from_state, surface_indices_from_static

    inputs = booz_xform_inputs_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
    )
    surface_indices, surface_values = surface_indices_from_static(static, [float(s)])
    constants, grids = prepare_booz_xform_constants_from_inputs(
        inputs=inputs,
        mboz=int(mboz),
        nboz=int(nboz),
        asym=bool(static.cfg.lasym),
    )
    out = booz_xform_from_inputs(
        inputs=inputs,
        constants=constants,
        grids=grids,
        surface_indices=jnp.asarray(surface_indices, dtype=jnp.int32),
        jit=True,
    )

    bmnc_b = jnp.asarray(out["bmnc_b"])[0]
    ixm_b = jnp.asarray(out["ixm_b"], dtype=jnp.int32)
    ixn_b = jnp.asarray(out["ixn_b"], dtype=jnp.int32)
    b0 = bmnc_b[0]
    include = jnp.abs(bmnc_b / b0) >= float(min_bmn_to_load)
    include = include.at[0].set(True)
    return BoozerSurface(
        m=ixm_b[include],
        n=jnp.asarray(jnp.rint(ixn_b[include] / int(inputs.nfp)), dtype=jnp.int32),
        b_cos=bmnc_b[include],
        nfp=int(inputs.nfp),
        iota=float(jnp.asarray(out["iota_b"])[0]),
        psi_p=float(psi_p),
        b_theta=float(jnp.asarray(out["buco_b"])[0]),
        b_zeta=float(jnp.asarray(out["bvco_b"])[0]),
        b0=float(b0),
        source_path=Path(getattr(indata, "input_filename", "vmec_jax_state")).expanduser(),
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
