"""Boozer-transform helpers for in-memory ``vmec_jax`` workflows."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from ._checkout_paths import find_booz_xform_jax_root, find_vmec_jax_root


def _apply_boozer_sign_convention(
    *,
    iota,
    b_theta,
    b_zeta,
):
    """Match the right-handed Boozer convention used by file-backed loading."""

    iota_value = -jnp.asarray(iota)
    b_theta_value = -jnp.asarray(b_theta)
    b_zeta_value = jnp.asarray(b_zeta)
    sign = jnp.where((b_zeta_value + iota_value * b_theta_value) >= 0.0, 1.0, -1.0)
    return iota_value, sign * b_theta_value, sign * b_zeta_value


def _apply_boozer_sign_convention_profiles(*, iotaf, buco, bvco, gmnc_b):
    """Apply the right-handed Boozer convention to radial-profile arrays."""

    iotaf_arr = jnp.asarray(iotaf)
    buco_arr = jnp.asarray(buco)
    bvco_arr = jnp.asarray(bvco)
    gmnc_arr = jnp.asarray(gmnc_b)

    iota_value = -iotaf_arr[1:]
    b_theta_value = -buco_arr[1:]
    b_zeta_value = bvco_arr[1:]
    sign = jnp.where((b_zeta_value + iota_value * b_theta_value) >= 0.0, 1.0, -1.0)

    return (
        jnp.concatenate([jnp.zeros((1,), dtype=iotaf_arr.dtype), iota_value], axis=0),
        jnp.concatenate(
            [jnp.zeros((1,), dtype=buco_arr.dtype), sign * b_theta_value],
            axis=0,
        ),
        jnp.concatenate(
            [jnp.zeros((1,), dtype=bvco_arr.dtype), sign * b_zeta_value],
            axis=0,
        ),
        sign[:, None] * gmnc_arr,
    )


def _prepend_checkout(root: Path | None) -> None:
    if root is None:
        return
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _import_vmec_jax():
    if "vmec_jax" in sys.modules:
        return sys.modules["vmec_jax"]
    try:
        import vmec_jax
    except ModuleNotFoundError:
        _prepend_checkout(find_vmec_jax_root())
        import vmec_jax

    return vmec_jax


def _import_booz_xform_jax_api():
    if "booz_xform_jax.jax_api" in sys.modules:
        return sys.modules["booz_xform_jax.jax_api"]
    try:
        from booz_xform_jax import jax_api
    except ModuleNotFoundError:
        _prepend_checkout(find_booz_xform_jax_root())
        from booz_xform_jax import jax_api

    return jax_api


def _booz_xform_bundle_from_vmec_jax_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    s_values: Sequence[float] | None,
    mboz: int,
    nboz: int,
    flux_profiles=None,
    profiles_half=None,
):
    vmec_jax = _import_vmec_jax()
    jax_api = _import_booz_xform_jax_api()
    legacy_builder = getattr(vmec_jax, "booz_xform_inputs_from_state", None)
    if legacy_builder is not None:
        inputs = legacy_builder(
            state=state,
            static=static,
            indata=indata,
            signgs=signgs,
            flux=flux_profiles,
            profiles_half=profiles_half,
        )
        surface_indices = None
        if s_values is not None:
            surface_indices, _surface_values = vmec_jax.surface_indices_from_static(
                static,
                [float(s_value) for s_value in s_values],
            )
        asym = bool(static.cfg.lasym)
    else:
        runtime_signgs = int(static.setup.signgs)
        if int(signgs) != runtime_signgs:
            raise ValueError(
                "signgs does not match the supplied vmec_jax SolverRuntime: "
                f"{signgs} != {runtime_signgs}"
            )
        inputs = _core_boozer_inputs_from_state(
            vmec_jax=vmec_jax,
            state=state,
            runtime=static,
            s_values=s_values,
        )
        surface_indices = None
        asym = bool(static.resolution.lasym)
    constants, grids = jax_api.prepare_booz_xform_constants_from_inputs(
        inputs=inputs,
        mboz=int(mboz),
        nboz=int(nboz),
        asym=asym,
    )
    out = jax_api.booz_xform_from_inputs(
        inputs=inputs,
        constants=constants,
        grids=grids,
        surface_indices=None
        if surface_indices is None
        else jnp.asarray(surface_indices, dtype=jnp.int32),
        jit=True,
    )
    return inputs, out


def _core_boozer_inputs_from_state(*, vmec_jax, state, runtime, s_values):
    """Adapt the current vmec_jax core tables to booz_xform_jax inputs."""

    try:
        from vmec_jax.core.boozer_tables import boozer_input_tables
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "This NTX workflow requires the current vmec_jax core API, "
            "including vmec_jax.core.boozer_tables.boozer_input_tables."
        ) from exc

    if not hasattr(runtime, "resolution") or not hasattr(runtime, "setup"):
        raise TypeError(
            "With the current vmec_jax API, `static` must be the matching "
            "vmec_jax SolverRuntime returned by prepare_runtime() or "
            "implicit.runtime_from_params()."
        )
    if bool(runtime.resolution.lasym):
        raise NotImplementedError(
            "The current differentiable vmec_jax Boozer table bridge supports "
            "stellarator-symmetric equilibria only."
        )

    s_full = jnp.asarray(runtime.setup.s_full)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])
    requested = s_half if s_values is None else jnp.asarray(s_values, dtype=s_half.dtype)
    if requested.ndim != 1 or int(requested.size) == 0:
        raise ValueError("s_values must contain at least one normalized-flux surface")
    if bool(jnp.any((requested < 0.0) | (requested > 1.0))):
        raise ValueError("s_values must lie between 0 and 1")

    # boozer_input_tables uses VMEC half-mesh row j, with 1 <= j < ns.
    rows = [int(jnp.argmin(jnp.abs(s_half - value))) + 1 for value in requested]
    tables = [boozer_input_tables(state, runtime, row) for row in rows]
    mode_m = tables[0]["xm"]
    mode_n = tables[0]["xn"]

    def stack(name):
        return jnp.stack([jnp.asarray(table[name]) for table in tables])

    return SimpleNamespace(
        nfp=int(runtime.resolution.nfp),
        xm=mode_m,
        xn=mode_n,
        xm_nyq=mode_m,
        xn_nyq=mode_n,
        rmnc=stack("rmnc"),
        zmns=stack("zmns"),
        lmns=stack("lmns"),
        bmnc=stack("bmnc"),
        bsubumnc=stack("bsubumnc"),
        bsubvmnc=stack("bsubvmnc"),
        iota=stack("iota"),
        rmns=None,
        zmnc=None,
        lmnc=None,
        bmns=None,
        bsubumns=None,
        bsubvmns=None,
    )


def _booz_xform_gmnc_from_inputs(*, inputs, mboz: int, nboz: int, asym: bool):
    jax_api = _import_booz_xform_jax_api()
    if not hasattr(jax_api, "_surface_transform") or not hasattr(jax_api, "_init_trig"):
        raise RuntimeError("booz_xform_jax internal JAX helpers are unavailable")

    constants, grids = jax_api.prepare_booz_xform_constants_from_inputs(
        inputs=inputs,
        mboz=int(mboz),
        nboz=int(nboz),
        asym=bool(asym),
    )

    xm_non = jnp.asarray(inputs.xm, dtype=jnp.int32)
    xn_non = jnp.asarray(inputs.xn, dtype=jnp.int32)
    xm_nyq = jnp.asarray(inputs.xm_nyq, dtype=jnp.int32)
    xn_nyq = jnp.asarray(inputs.xn_nyq, dtype=jnp.int32)

    cosm, sinm, cosn, sinn = jax_api._init_trig(
        grids.theta_grid,
        grids.zeta_grid,
        constants.mmax_non,
        constants.nmax_non,
        constants.nfp,
    )
    cosm_nyq, sinm_nyq, cosn_nyq, sinn_nyq = jax_api._init_trig(
        grids.theta_grid,
        grids.zeta_grid,
        constants.mmax_nyq,
        constants.nmax_nyq,
        constants.nfp,
    )

    cosm_m_non = jnp.take(cosm, xm_non, axis=1)
    sinm_m_non = jnp.take(sinm, xm_non, axis=1)
    abs_n_non = jnp.abs(xn_non // constants.nfp)
    cosn_n_non = jnp.take(cosn, abs_n_non, axis=1)
    sinn_n_non = jnp.take(sinn, abs_n_non, axis=1)
    sign_non = jnp.where(xn_non < 0, -1.0, 1.0)[None, :]
    tcos_non = cosm_m_non * cosn_n_non + sinm_m_non * sinn_n_non * sign_non
    tsin_non = sinm_m_non * cosn_n_non - cosm_m_non * sinn_n_non * sign_non

    cosm_m_nyq = jnp.take(cosm_nyq, xm_nyq, axis=1)
    sinm_m_nyq = jnp.take(sinm_nyq, xm_nyq, axis=1)
    abs_n_nyq = jnp.abs(xn_nyq // constants.nfp)
    cosn_n_nyq = jnp.take(cosn_nyq, abs_n_nyq, axis=1)
    sinn_n_nyq = jnp.take(sinn_nyq, abs_n_nyq, axis=1)
    sign_nyq = jnp.where(xn_nyq < 0, -1.0, 1.0)[None, :]
    tcos_nyq = cosm_m_nyq * cosn_n_nyq + sinm_m_nyq * sinn_n_nyq * sign_nyq
    tsin_nyq = sinm_m_nyq * cosn_n_nyq - cosm_m_nyq * sinn_n_nyq * sign_nyq

    m_non_f = xm_non.astype(jnp.float64)
    n_non_f = xn_non.astype(jnp.float64)
    m_nyq_f = xm_nyq.astype(jnp.float64)
    n_nyq_f = xn_nyq.astype(jnp.float64)
    idx_theta0 = jnp.arange(0, constants.nzeta)
    idx_thetapi = jnp.arange(
        (constants.nu2_b - 1) * constants.nzeta,
        constants.nu2_b * constants.nzeta,
    )
    m_b = grids.xm_b
    abs_n_b = jnp.abs(grids.xn_b // constants.nfp)
    sign_b = jnp.where(grids.xn_b < 0, -1.0, 1.0)[None, :]

    def surface_transform(
        rmnc,
        zmns,
        lmns,
        bmnc,
        bsubumnc,
        bsubvmnc,
        iota,
        bmns,
        bsubumns,
        bsubvmns,
    ):
        return jax_api._surface_transform(
            rmnc,
            zmns,
            lmns,
            bmnc,
            bsubumnc,
            bsubvmnc,
            iota,
            constants=constants,
            grids=grids,
            tcos_non=tcos_non,
            tsin_non=tsin_non,
            tcos_nyq=tcos_nyq,
            tsin_nyq=tsin_nyq,
            m_non_f=m_non_f,
            n_non_f=n_non_f,
            m_nyq_f=m_nyq_f,
            n_nyq_f=n_nyq_f,
            idx_theta0=idx_theta0,
            idx_thetapi=idx_thetapi,
            m_b=m_b,
            abs_n_b=abs_n_b,
            sign_b=sign_b,
            bmns=bmns,
            bsubumns=bsubumns,
            bsubvmns=bsubvmns,
            fourier_mode="vectorized",
            trig_f32=False,
        )

    bmns_in = inputs.bmns if inputs.bmns is not None else jnp.zeros_like(inputs.bmnc)
    bsubumns_in = (
        inputs.bsubumns if inputs.bsubumns is not None else jnp.zeros_like(inputs.bsubumnc)
    )
    bsubvmns_in = (
        inputs.bsubvmns if inputs.bsubvmns is not None else jnp.zeros_like(inputs.bsubvmnc)
    )
    outputs = jax.vmap(surface_transform)(
        inputs.rmnc,
        inputs.zmns,
        inputs.lmns,
        inputs.bmnc,
        inputs.bsubumnc,
        inputs.bsubvmnc,
        inputs.iota,
        bmns_in,
        bsubumns_in,
        bsubvmns_in,
    )
    return outputs[4]
