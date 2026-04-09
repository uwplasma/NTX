"""Helpers for `vmec_jax -> booz_xform_jax -> NTX` workflows."""

from __future__ import annotations

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
