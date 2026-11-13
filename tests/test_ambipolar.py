from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ntx.ambipolar as amb
from ntx import surface_from_vmex_vmec_wout_file
from ntx.grids import GridSpec

SAMPLE_WOUT = Path(__file__).parent / "fixtures" / "sample_wout.nc"


def _plasma():
    return amb.AmbipolarPlasma(
        density=lambda r: 2.0e19 * (1.0 - r**2) + 1.0e18,
        te=lambda r: 3000.0 * (1.0 - r**2) + 100.0,
        ti=lambda r: 1000.0 * (1.0 - r**2) + 100.0,
    )


def _small_root(**kwargs):
    return amb.ElectronRoot(
        [0.5],
        _plasma(),
        1.0,
        e_max=2.0e4,
        n_energy=4,
        n_nu=3,
        n_field=3,
        n_er=41,
        grid=GridSpec(5, 5, 4),
        mboz=4,
        nboz=4,
        **kwargs,
    )


def _synthetic_tables(root, tilt):
    """Smooth ``log10 D11`` table whose field dependence is set by ``tilt``."""
    nu = jnp.linspace(-1.0, 1.0, 3)[:, None]
    e = jnp.linspace(0.0, 1.0, 3)[None, :]
    return (-3.0 + 0.2 * nu + tilt * e)[None]


def test_ambipolar_roots_finds_every_root_with_implicit_derivative():
    def residual(e, d):
        return (e + 1.0) * (e - d[0]) * (e - 3.0)

    grid = jnp.linspace(-2.03, 4.01, 13)
    roots, valid, stable = amb.ambipolar_roots(residual, grid, jnp.asarray([0.5]))
    assert int(valid.sum()) == 3
    np.testing.assert_allclose(np.sort(np.asarray(roots)[np.asarray(valid)]), [-1.0, 0.5, 3.0])
    assert [bool(s) for s, v in zip(stable, valid, strict=True) if v] == [True, False, True]

    def middle_root(d):
        return amb.ambipolar_roots(residual, grid, d)[0][np.flatnonzero(np.asarray(valid))[1]]

    grad = jax.grad(middle_root)(jnp.asarray([0.5]))
    np.testing.assert_allclose(np.asarray(grad), [1.0], atol=1e-6)


def test_electron_root_flux_roots_and_hinge(monkeypatch):
    root = _small_root(e_target=0.0)
    tables = _synthetic_tables(root, 0.3)
    monkeypatch.setattr(root, "tables", lambda state, runtime: tables)
    flux = root.flux(jnp.asarray([0.0, 100.0]), tables, 0)
    assert flux.shape == (2, 2) and bool(jnp.all(jnp.isfinite(flux)))
    roots, valid, stable = root.roots(tables)
    assert roots.shape == valid.shape == stable.shape == (1, 40)
    top = float(root.top_root(tables)[0])
    assert np.isfinite(top)
    assert abs(float(root.residual(jnp.asarray(top), tables, 0))) < 1e-6 * float(
        jnp.sum(jnp.abs(root.flux(jnp.asarray(top), tables, 0)))
    )
    hinge = root(None, None)
    assert hinge.shape == (1,) and float(hinge[0]) >= 0.0

    def root_of_tilt(c):
        t = _synthetic_tables(root, c)
        return root.top_root(t)[0]

    h = 1.0e-4
    fd = (root_of_tilt(0.3 + h) - root_of_tilt(0.3 - h)) / (2.0 * h)
    np.testing.assert_allclose(float(jax.grad(root_of_tilt)(0.3)), float(fd), rtol=1e-4)


def test_electron_root_tables_use_vmex_surfaces(monkeypatch):
    root = _small_root()
    calls = {}

    def fake_surfaces(**kwargs):
        calls.update(kwargs)
        return (surface_from_vmex_vmec_wout_file(SAMPLE_WOUT, s=0.25),)

    monkeypatch.setattr(amb, "surfaces_from_vmex_state", fake_surfaces)
    runtime = SimpleNamespace(setup=SimpleNamespace(signgs=-1))
    tables = root.tables(object(), runtime)
    assert tables.shape == (1, 3, 3) and bool(jnp.all(jnp.isfinite(tables)))
    assert calls["s_values"] == (0.25,) and calls["signgs"] == -1


@pytest.mark.skipif(
    os.environ.get("NTX_RUN_BOUNDARY_AUTODIFF") != "1",
    reason="needs vmex; the reverse-mode compile takes minutes (NTX_RUN_BOUNDARY_AUTODIFF=1)",
)
def test_top_root_gradient_matches_central_differences_through_vmex():
    """d(top root)/d(boundary) through VMEX, Boozer and NTX against central differences."""
    from dataclasses import replace

    import vmex as vj
    from vmex import optimize as opt

    deck = Path(vj.__file__).parent.parent / "examples" / "data" / "input.QI_nfp2_initial"
    inp = replace(
        vj.VmecInput.from_file(deck),
        ns_array=np.array([31]),
        ftol_array=np.array([1e-12]),
        niter_array=np.array([5500]),
        delt=0.5,
    ).change_resolution(mpol=5, ntor=5, ntheta=16, nzeta=14)
    phiedge = float(opt.solve_equilibrium(inp).wout.phi[-1])
    plasma = amb.AmbipolarPlasma(
        density=lambda r: 1e18 * (1 - r**2) + 1e17,
        te=lambda r: 8000 * (1 - r**2) ** 1.5 + 100,
        ti=lambda r: 1000 * (1 - r**2) + 100,
    )
    root = amb.ElectronRoot([0.6], plasma, phiedge, e_max=3.0e4)

    def top_root_kv(state, runtime):
        return root.top_root(root.tables(state, runtime)) / 1.0e3

    problem = opt.VmecProblem.from_tuples(inp, [(top_root_kv, 0.0, 1.0)], max_mode=1, use_ess=False)
    x0 = problem.x0
    jac = np.asarray(problem.residual_jac(x0))[0]
    for h, tol in ((1.0e-3, 5.0e-2), (1.0e-4, 2.0e-3)):
        for k in range(len(x0)):
            step = np.zeros_like(x0)
            step[k] = h * problem.scales[k]
            fd = (problem.residual(x0 + step)[0] - problem.residual(x0 - step)[0]) / (2 * step[k])
            assert abs(fd - jac[k]) <= tol * np.max(np.abs(jac)), (h, k, fd, jac[k])
