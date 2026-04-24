"""Boozer-transform helpers for VMEC-JAX-backed NEOPAX field builders."""

from __future__ import annotations

from ._vmec_jax_boozer import (
    _booz_xform_bundle_from_vmec_jax_state,
    _booz_xform_gmnc_from_inputs,
)


def _booz_xform_bundle_with_gmnc_from_vmec_jax_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    mboz: int,
    nboz: int,
):
    inputs, out = _booz_xform_bundle_from_vmec_jax_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        s_values=None,
        mboz=mboz,
        nboz=nboz,
    )
    gmnc_b = _booz_xform_gmnc_from_inputs(
        inputs=inputs,
        mboz=mboz,
        nboz=nboz,
        asym=bool(static.cfg.lasym),
    )
    out_with_gmnc = dict(out)
    out_with_gmnc["gmnc_b"] = gmnc_b
    return inputs, out_with_gmnc


__all__ = [
    "_booz_xform_bundle_with_gmnc_from_vmec_jax_state",
    "_booz_xform_gmnc_from_inputs",
]
