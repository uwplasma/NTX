"""VMEC-JAX scalar and radial-profile helpers for NEOPAX field builders."""

from __future__ import annotations

import jax.numpy as jnp


def _vmec_s_full(static):
    setup = getattr(static, "setup", None)
    return jnp.asarray(setup.s_full if setup is not None else static.s)


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
    setup = getattr(static, "setup", None)
    if setup is not None:
        s_full = jnp.asarray(setup.s_full)
        phipf = jnp.asarray(setup.phipf)
        return jnp.abs(jnp.sum(phipf[1:] * jnp.diff(s_full)))

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
    rcos = getattr(state, "R_cos", getattr(state, "Rcos", None))
    if rcos is None:
        raise AttributeError("vmec_jax state does not expose `R_cos` or `Rcos`")
    return jnp.asarray(rcos)[-1, 0]


def _vmec_psia_from_indata(*, indata, static, signgs: int):
    if hasattr(indata, "phiedge"):
        return jnp.abs(jnp.asarray(indata.phiedge) / (2.0 * jnp.pi))

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
    if hasattr(static, "setup") and hasattr(state, "R_cos"):
        from vmec_jax.core.fields import (
            energies_and_force_norms,
            magnetic_fields,
            metric_elements,
        )
        from vmec_jax.core.geometry import half_mesh_jacobian
        from vmec_jax.core.solver import _geometry

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
    "_vmec_s_full",
    "_vmec_volume_profiles_from_state",
]
