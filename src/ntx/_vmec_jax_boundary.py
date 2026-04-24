"""Boundary-state helpers for optional ``vmec_jax`` workflows."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from ._vmec_jax_boozer import _import_vmec_jax


@dataclasses.dataclass(frozen=True)
class VmecJaxBoundaryContext:
    """Static inputs for differentiable fixed-boundary VMEC workflows."""

    input_path: Path
    cfg: Any
    indata: Any
    static: Any
    signgs: int
    boundary: Any
    specs: tuple[object, ...]


def build_vmec_jax_boundary_context(
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

    vmec_jax = _import_vmec_jax()
    vmec_input = Path(input_path).expanduser().resolve()
    cfg, indata = vmec_jax.load_config(vmec_input)
    static = vmec_jax.build_static(cfg)
    boundary = vmec_jax.boundary_input_from_indata(indata, static.modes)
    specs = vmec_jax.boundary_param_specs(
        boundary,
        static.modes,
        max_mode=max_mode,
        include=include,
        fix=fix,
        include_axis=include_axis,
    )
    return VmecJaxBoundaryContext(
        input_path=vmec_input,
        cfg=cfg,
        indata=indata,
        static=static,
        signgs=int(signgs),
        boundary=boundary,
        specs=tuple(specs),
    )


def initial_guess_vmec_jax_boundary_state(
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

    vmec_jax = _import_vmec_jax()
    boundary = vmec_jax.apply_boundary_params(context.boundary, context.specs, params)
    return vmec_jax.initial_guess_from_boundary(
        context.static,
        boundary,
        context.indata,
        vmec_project=vmec_project,
    )


def solve_vmec_jax_boundary_state(
    context: VmecJaxBoundaryContext,
    params,
    *,
    vmec_project: bool = True,
    max_iter: int = 50,
    step_size: float = 1.0,
    ftol: float | None = None,
    implicit=None,
):
    """Solve a fixed-boundary `vmec_jax` state from traced boundary parameters.

    The explicit edge arrays are passed into the implicit VMEC residual solve so
    the boundary dependence is preserved through the stop-gradient initial guess
    used inside `vmec_jax`.
    """

    vmec_jax = _import_vmec_jax()
    state0 = initial_guess_vmec_jax_boundary_state(
        context,
        params,
        vmec_project=vmec_project,
    )
    return vmec_jax.implicit.solve_fixed_boundary_state_implicit_vmec_residual(
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


def relax_vmec_jax_boundary_state_explicit(
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
    """Run the explicit fixed-step `vmec_jax` boundary relaxation.

    This is the forward-mode boundary-to-output lane that keeps the equilibrium
    dependence inside an unrolled JAX-compatible solve. It is intentionally
    separate from the implicit VMEC helper: this routine exposes the explicit
    update path used by the self-consistent boundary-control derivative audit,
    while the implicit helper exposes the upstream custom-VJP solve.
    """

    vmec_jax = _import_vmec_jax()
    state0 = initial_guess_vmec_jax_boundary_state(
        context,
        params,
        vmec_project=vmec_project,
    )
    flux = vmec_jax.flux_profiles_from_indata(
        context.indata,
        context.static.s,
        signgs=context.signgs,
    )
    pressure_value = (
        jnp.zeros_like(jnp.asarray(context.static.s))
        if pressure is None
        else jnp.asarray(pressure)
    )
    result = vmec_jax.solve_fixed_boundary_gd(
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
    "build_vmec_jax_boundary_context",
    "initial_guess_vmec_jax_boundary_state",
    "relax_vmec_jax_boundary_state_explicit",
    "solve_vmec_jax_boundary_state",
]
