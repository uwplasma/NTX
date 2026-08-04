"""Reading VMEX equilibria: surfaces, boundary, and Boozer transform.

Everything NTX needs from a VMEX output file, from the flux surfaces and
boundary through to the Boozer-coordinate transform.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._checkout_paths import find_booz_xform_jax_root, find_vmex_root
from .geometry import BoozerSurface

__all__ = [
    "VmecJaxBoundaryContext",
    "build_vmex_boundary_context",
    "initial_guess_vmex_boundary_state",
    "relax_vmex_boundary_state_explicit",
    "solve_vmex_boundary_state",
    "surface_from_vmex_state",
    "surface_from_vmex_wout",
    "surfaces_from_vmex_boundary_params",
    "surfaces_from_vmex_state",
]


# --- _vmex_boozer: Boozer-transform helpers for in-memory ``vmex`` workflows. ---


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
    """Put a sibling checkout at the front of sys.path.

    Front rather than back, so a local checkout under development wins over an
    installed copy of the same package.
    """
    if root is None:
        return
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _import_vmex():
    """Import vmex, from a sibling checkout if it is not installed.

    Returns the cached module when already imported, so path manipulation
    happens at most once per process.
    """
    if "vmex" in sys.modules:
        return sys.modules["vmex"]
    try:
        import vmex
    except ModuleNotFoundError:
        _prepend_checkout(find_vmex_root())
        import vmex

    return vmex


def _import_booz_xform_jax_api():
    """Import the booz_xform_jax JAX API, from a sibling checkout if needed."""
    if "booz_xform_jax.jax_api" in sys.modules:
        return sys.modules["booz_xform_jax.jax_api"]
    try:
        from booz_xform_jax import jax_api
    except ModuleNotFoundError:
        _prepend_checkout(find_booz_xform_jax_root())
        from booz_xform_jax import jax_api

    return jax_api


def _booz_xform_bundle_from_vmex_state(
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
    """Build Boozer-transform inputs and outputs from a VMEX equilibrium state.

    Prefers vmex's own builder when the installed version provides it and falls
    back to assembling the inputs here, so both older and newer vmex work.
    """
    vmex = _import_vmex()
    jax_api = _import_booz_xform_jax_api()
    legacy_builder = getattr(vmex, "booz_xform_inputs_from_state", None)
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
            surface_indices, _surface_values = vmex.surface_indices_from_static(
                static,
                [float(s_value) for s_value in s_values],
            )
        asym = bool(static.cfg.lasym)
    else:
        runtime_signgs = int(static.setup.signgs)
        if int(signgs) != runtime_signgs:
            raise ValueError(
                "signgs does not match the supplied vmex SolverRuntime: "
                f"{signgs} != {runtime_signgs}"
            )
        inputs = _core_boozer_inputs_from_state(
            vmex=vmex,
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


def _core_boozer_inputs_from_state(*, vmex, state, runtime, s_values):
    """Adapt the current vmex core tables to booz_xform_jax inputs."""

    try:
        from vmex.core.boozer_tables import boozer_input_tables
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "This NTX workflow requires the current vmex core API, "
            "including vmex.core.boozer_tables.boozer_input_tables."
        ) from exc

    if not hasattr(runtime, "resolution") or not hasattr(runtime, "setup"):
        raise TypeError(
            "With the current vmex API, `static` must be the matching "
            "vmex SolverRuntime returned by prepare_runtime() or "
            "implicit.runtime_from_params()."
        )
    if bool(runtime.resolution.lasym):
        raise NotImplementedError(
            "The current differentiable vmex Boozer table bridge supports "
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
    """Compute the Jacobian harmonics gmnc from prepared Boozer inputs.

    Reaches for two private booz_xform_jax helpers and raises a clear error when
    they are absent, rather than failing later with an attribute error deep in
    the transform.
    """
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


# --- _vmex_boundary: Boundary-state helpers for optional ``vmex`` workflows. ---


@dataclasses.dataclass(frozen=True)
class VmecJaxBoundaryContext:
    """Static inputs for differentiable fixed-boundary VMEC workflows."""

    input_path: Path
    cfg: Any
    indata: Any
    static: Any
    signgs: int
    boundary: Any
    specs: tuple[Any, ...]
    backend: str = "legacy"


@dataclasses.dataclass(frozen=True)
class _CoreBoundaryParamSpec:
    """One additive boundary coefficient in the current vmex basis."""

    name: str
    field: str
    n_index: int
    m_index: int


def build_vmex_boundary_context(
    input_path: str | Path,
    *,
    signgs: int = 1,
    max_mode: int | None = 1,
    include: Sequence[str] = ("rc", "zs"),
    fix: Sequence[str] = ("rc00",),
    include_axis: bool = False,
) -> VmecJaxBoundaryContext:
    """Bundle the static inputs needed for differentiable boundary scans.

    The returned context is intended to be closed over by JAX objectives while
    only the boundary-parameter vector is traced.
    """

    vmex = _import_vmex()
    vmec_input = Path(input_path).expanduser().resolve()
    if hasattr(vmex, "VmecInput"):
        indata = vmex.VmecInput.from_file(str(vmec_input))
        cfg = vmex.implicit.make_config(indata)
        boundary = vmex.implicit.params_from_input(indata)
        static = vmex.implicit.runtime_from_params(boundary, cfg)
        specs = _core_boundary_param_specs(
            indata,
            max_mode=max_mode,
            include=include,
            fix=fix,
            include_axis=include_axis,
        )
        backend = "core"
        signgs = int(static.setup.signgs)
    else:
        cfg, indata = vmex.load_config(vmec_input)
        static = vmex.build_static(cfg)
        boundary = vmex.boundary_input_from_indata(indata, static.modes)
        specs = vmex.boundary_param_specs(
            boundary,
            static.modes,
            max_mode=max_mode,
            include=include,
            fix=fix,
            include_axis=include_axis,
        )
        backend = "legacy"
    return VmecJaxBoundaryContext(
        input_path=vmec_input,
        cfg=cfg,
        indata=indata,
        static=static,
        signgs=int(signgs),
        boundary=boundary,
        specs=tuple(specs),
        backend=backend,
    )


def _core_boundary_param_specs(
    inp,
    *,
    max_mode: int | None,
    include: Sequence[str],
    fix: Sequence[str],
    include_axis: bool,
) -> tuple[_CoreBoundaryParamSpec, ...]:
    """Enumerate the boundary coefficients exposed as optimization parameters.

    Honours an include/fix split and a mode-number cutoff, so a design study can
    vary a chosen subset of the boundary while holding the rest fixed.
    """
    include_set = {name.lower() for name in include}
    fix_set = {name.lower() for name in fix}
    families = (("rc", "rbc"), ("rs", "rbs"), ("zc", "zbc"), ("zs", "zbs"))
    specs = []
    for m in range(int(inp.mpol)):
        for n in range(-int(inp.ntor), int(inp.ntor) + 1):
            if m == 0 and n < 0:
                continue
            if max_mode is not None and (m > int(max_mode) or abs(n) > int(max_mode)):
                continue
            if not include_axis and m == 0 and n == 0:
                continue
            suffix = f"{m}{n:+d}".replace("+", "")
            for prefix, field in families:
                name = f"{prefix}{suffix}"
                if prefix in include_set and name.lower() not in fix_set:
                    specs.append(
                        _CoreBoundaryParamSpec(
                            name=name,
                            field=field,
                            n_index=n + int(inp.ntor),
                            m_index=m,
                        )
                    )
    return tuple(specs)


def _core_params_with_updates(context: VmecJaxBoundaryContext, params):
    """Apply a flat parameter vector back onto the boundary coefficients.

    Checks the length against the spec list first: a silently mismatched vector
    would scatter parameters onto the wrong harmonics.
    """
    values = jnp.asarray(params)
    if values.ndim != 1 or int(values.size) != len(context.specs):
        raise ValueError(f"expected {len(context.specs)} boundary updates, got {values.size}")
    arrays = {
        name: jnp.asarray(getattr(context.boundary, name)) for name in ("rbc", "rbs", "zbc", "zbs")
    }
    for value, spec in zip(values, context.specs, strict=True):
        arrays[spec.field] = arrays[spec.field].at[spec.n_index, spec.m_index].add(value)
    return dataclasses.replace(context.boundary, **arrays)


def initial_guess_vmex_boundary_state(
    context: VmecJaxBoundaryContext,
    params,
    *,
    vmec_project: bool = False,
):
    """Build the differentiable boundary-projected VMEC state before relaxation.

    This helper is the low-dimensional forward-mode boundary-control lane used
    by the boundary-to-output derivative benchmarks. It preserves the
    differentiable map from boundary coefficients to the projected VMEC state
    without invoking the implicit equilibrium solve.
    """

    vmex = _import_vmex()
    if context.backend == "core":
        from vmex.core.solver import _initial_state

        updated = _core_params_with_updates(context, params)
        runtime = vmex.implicit.runtime_from_params(updated, context.cfg)
        return _initial_state(runtime.setup)
    boundary = vmex.apply_boundary_params(context.boundary, context.specs, params)
    return vmex.initial_guess_from_boundary(
        context.static,
        boundary,
        context.indata,
        vmec_project=vmec_project,
    )


def solve_vmex_boundary_state(
    context: VmecJaxBoundaryContext,
    params,
    *,
    vmec_project: bool = True,
    max_iter: int = 50,
    step_size: float = 1.0,
    ftol: float | None = None,
    implicit=None,
):
    """Solve a fixed-boundary `vmex` state from traced boundary parameters.

    The explicit edge arrays are passed into the implicit VMEC residual solve so
    the boundary dependence is preserved through the stop-gradient initial guess
    used inside `vmex`.
    """

    vmex = _import_vmex()
    if context.backend == "core":
        updated = _core_params_with_updates(context, params)
        cfg = context.cfg
        replacements: dict[str, int | float] = {}
        if max_iter is not None:
            # The current implicit API returns only converged roots. Legacy
            # callers used small values for truncated explicit smoke runs, so
            # never reduce the equilibrium deck's convergence budget here.
            replacements["max_iterations"] = max(int(cfg.max_iterations), int(max_iter))
        if ftol is not None:
            replacements["ftol"] = float(ftol)
        if replacements:
            cfg = dataclasses.replace(cfg, **replacements)
        return vmex.implicit.solve_implicit(updated, cfg)
    state0 = initial_guess_vmex_boundary_state(
        context,
        params,
        vmec_project=vmec_project,
    )
    return vmex.implicit.solve_fixed_boundary_state_implicit_vmec_residual(
        state0,
        context.static,
        indata=context.indata,
        signgs=context.signgs,
        max_iter=max_iter,
        step_size=step_size,
        ftol=ftol,
        implicit=implicit,
        edge_Rcos=state0.Rcos[-1, :],
        edge_Rsin=state0.Rsin[-1, :],
        edge_Zcos=state0.Zcos[-1, :],
        edge_Zsin=state0.Zsin[-1, :],
    )


def relax_vmex_boundary_state_explicit(
    context: VmecJaxBoundaryContext,
    params,
    *,
    vmec_project: bool = False,
    max_iter: int = 10,
    step_size: float = 1.0e-8,
    pressure=None,
    jacobian_penalty: float = 1.0e3,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    stop_grad_in_update: bool = False,
    differentiable: bool = True,
    verbose: bool = False,
):
    """Run the explicit fixed-step `vmex` boundary relaxation.

    This is the forward-mode boundary-to-output lane that keeps the equilibrium
    dependence inside an unrolled JAX-compatible solve. It is intentionally
    separate from the implicit VMEC helper: this routine exposes the explicit
    update path used by the self-consistent boundary-control derivative audit,
    while the implicit helper exposes the upstream custom-VJP solve.
    """

    vmex = _import_vmex()
    if context.backend == "core":
        raise NotImplementedError(
            "The current vmex API removed the experimental explicit "
            "fixed-step relaxation. Use solve_vmex_boundary_state(), "
            "which uses vmex.implicit.solve_implicit and its validated "
            "custom-VJP equilibrium derivative."
        )
    state0 = initial_guess_vmex_boundary_state(
        context,
        params,
        vmec_project=vmec_project,
    )
    flux = vmex.flux_profiles_from_indata(
        context.indata,
        context.static.s,
        signgs=context.signgs,
    )
    pressure_value = (
        jnp.zeros_like(jnp.asarray(context.static.s)) if pressure is None else jnp.asarray(pressure)
    )
    result = vmex.solve_fixed_boundary_gd(
        state0,
        context.static,
        phipf=flux.phipf,
        chipf=flux.chipf,
        signgs=context.signgs,
        lamscale=flux.lamscale,
        pressure=pressure_value,
        gamma=float(context.indata.get_float("GAMMA", 0.0)),
        jacobian_penalty=float(jacobian_penalty),
        max_iter=int(max_iter),
        step_size=float(step_size),
        preconditioner=preconditioner,
        precond_exponent=float(precond_exponent),
        precond_radial_alpha=float(precond_radial_alpha),
        differentiable=bool(differentiable),
        stop_grad_in_update=bool(stop_grad_in_update),
        verbose=bool(verbose),
        edge_Rcos=state0.Rcos[-1, :],
        edge_Rsin=state0.Rsin[-1, :],
        edge_Zcos=state0.Zcos[-1, :],
        edge_Zsin=state0.Zsin[-1, :],
    )
    return result.state


# --- _vmex_surfaces: Boozer-surface builders for optional ``vmex`` workflows. ---


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
                "surface_from_vmex_wout requires vmex. Install it with `pip install vmex`."
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
