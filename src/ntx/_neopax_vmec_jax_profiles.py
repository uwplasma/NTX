"""VMEC-JAX scalar and radial-profile helpers for NEOPAX field builders."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp


def _rho_half_mesh_from_s(s_full):
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
    phipf_out = getattr(state, "phipf_out", None)
    if phipf_out is None:
        raise AttributeError("vmec_jax state does not expose `phipf_out`")

    try:
        from vmec_jax.integrals import cumrect_s_halfmesh
    except ModuleNotFoundError as exc:  # pragma: no cover - local checkout fallback
        raise ImportError("vmec_jax is required for differentiable field builders") from exc

    phi = cumrect_s_halfmesh(jnp.asarray(phipf_out), jnp.asarray(static.s))
    return jnp.abs(phi[-1])


def _vmec_edge_r00_from_state(state):
    rcos = getattr(state, "Rcos", None)
    if rcos is None:
        raise AttributeError("vmec_jax state does not expose `Rcos`")
    return jnp.asarray(rcos)[-1, 0]


def _vmec_psia_from_indata(*, indata, static, signgs: int):
    try:
        from vmec_jax.energy import flux_profiles_from_indata
        from vmec_jax.integrals import cumrect_s_halfmesh
    except ModuleNotFoundError as exc:  # pragma: no cover - local checkout fallback
        raise ImportError("vmec_jax is required for differentiable field builders") from exc

    flux = flux_profiles_from_indata(indata, jnp.asarray(static.s), signgs=int(signgs))
    phipf_out = jnp.asarray(flux.phipf)
    phi = cumrect_s_halfmesh(jnp.asarray(phipf_out), jnp.asarray(static.s))
    return jnp.abs(phi[-1])


def _vmec_volume_profiles_from_state(*, state, static, indata, signgs: int):
    try:
        from vmec_jax.vmec_forces import vmec_forces_rz_from_wout
        from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar_dynamic
    except ModuleNotFoundError as exc:  # pragma: no cover - local checkout fallback
        raise ImportError("vmec_jax is required for differentiable field builders") from exc

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


__all__ = [
    "_rho_half_mesh_from_s",
    "_vmec_edge_r00_from_state",
    "_vmec_psia_from_indata",
    "_vmec_psia_from_state",
    "_vmec_volume_profiles_from_state",
]
