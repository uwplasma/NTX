"""Boundary-state helpers for optional ``vmex`` workflows."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from ._vmex_boozer import _import_vmex


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


__all__ = [
    "VmecJaxBoundaryContext",
    "build_vmex_boundary_context",
    "initial_guess_vmex_boundary_state",
    "relax_vmex_boundary_state_explicit",
    "solve_vmex_boundary_state",
]
