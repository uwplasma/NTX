"""Differentiable NEOPAX field builders backed by vmec_jax states."""

from __future__ import annotations

from types import SimpleNamespace

import interpax
import jax
import jax.numpy as jnp

from ._neopax_field_utils import (
    _safe_divide,
    _safe_reciprocal,
    _surface_b10,
    _surface_bsqav,
)
from ._neopax_types import DifferentiableNeopaxField
from .vmec_jax_backend import (
    VmecJaxBoundaryContext,
    _booz_xform_bundle_from_vmec_jax_state,
    _import_booz_xform_jax_api,
    solve_vmec_jax_boundary_state,
    surfaces_from_vmec_jax_state,
)


def build_differentiable_neopax_field_from_vmec_jax_state(
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
    """Build a tracer-safe NEOPAX field from an in-memory `vmec_jax` state."""

    s_full = jnp.asarray(static.s)
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
        0.5 * (rho_grid[0] + rho_grid[1])
        if n_r_int > 1
        else jnp.asarray(0.0, dtype=rho_grid.dtype)
    )
    rho_grid_half = jnp.linspace(rho_grid_half0, rho_grid_half0 + rho_grid[-1], n_r_int)
    r_grid = rho_grid * a_b
    if n_r_int > 1:
        r_grid = r_grid.at[0].set(0.5 * r_grid[1])
    r_grid_half = rho_grid_half * a_b
    dr = r_grid[2] - r_grid[1] if n_r_int > 2 else jnp.asarray(0.0, dtype=r_grid.dtype)

    dVdr = interpax.Interpolator1D(rho_half[1:], jnp.asarray(vp)[1:], extrap=True)
    vprime = dVdr(rho_grid) * 2.0 * rho_grid / a_b
    vprime_half = dVdr(rho_grid_half) * 2.0 * rho_grid_half / a_b
    over_vprime = _safe_reciprocal(vprime)
    over_vprime = over_vprime.at[0].set(0.0)

    sample_rho = rho_grid[1:-1]
    sample_surfaces = surfaces_from_vmec_jax_state(
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
    sqrtg00_interp = interpax.Interpolator1D(rho_grid, sqrtg00_full, extrap=True)
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


def build_differentiable_neopax_field_from_vmec_jax_boundary_params(
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

    state = solve_vmec_jax_boundary_state(
        context,
        params,
        vmec_project=vmec_project,
        max_iter=max_iter,
        step_size=step_size,
        ftol=ftol,
        implicit=implicit,
    )
    return build_differentiable_neopax_field_from_vmec_jax_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=context.signgs,
        n_r=n_r,
        mboz=mboz,
        nboz=nboz,
        apply_boozer_sign_convention=apply_boozer_sign_convention,
    )


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


def _apply_boozer_sign_convention_profiles(*, iotaf, buco, bvco, gmnc_b):
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
