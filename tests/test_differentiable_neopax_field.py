from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from ntx import (
    GridSpec,
    build_differentiable_neopax_field,
    build_differentiable_neopax_field_from_vmec_jax_state,
    build_ntx_neopax_scan_from_vmec_jax_state,
    build_vmec_jax_boundary_context,
    get_differentiable_neopax_fluxes,
    solve_vmec_jax_boundary_state,
    to_neopax_monoenergetic,
)
from ntx._checkout_paths import (
    find_booz_xform_jax_root,
    find_neopax_root,
    find_vmec_jax_example_input,
)
from ntx._geometry_types import BoozerSurface
from ntx._neopax_field import (
    _find_mode_index,
    _safe_divide,
    _safe_reciprocal,
    _surface_b10,
    _surface_bsqav,
)
from ntx._neopax_vmec_jax_field import (
    _apply_boozer_sign_convention_profiles,
    _booz_xform_bundle_with_gmnc_from_vmec_jax_state,
    _booz_xform_gmnc_from_inputs,
    _rho_half_mesh_from_s,
    _vmec_edge_r00_from_state,
    _vmec_psia_from_indata,
    _vmec_psia_from_state,
    _vmec_volume_profiles_from_state,
    build_differentiable_neopax_field_from_vmec_jax_boundary_params,
)


def _import_neopax():
    sys.modules.pop("NEOPAX", None)
    try:
        import NEOPAX
    except ModuleNotFoundError:
        root = find_neopax_root()
        if root is None:
            raise
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        import NEOPAX

    if not hasattr(NEOPAX, "Field"):
        from NEOPAX._field import Field

        NEOPAX.Field = Field
    if not hasattr(NEOPAX, "Grid"):
        from NEOPAX._grid import Grid

        NEOPAX.Grid = Grid
    if not hasattr(NEOPAX, "Species"):
        from NEOPAX._species import Species

        NEOPAX.Species = Species
    if not hasattr(NEOPAX, "get_Neoclassical_Fluxes"):
        from NEOPAX._neoclassical import get_Neoclassical_Fluxes

        NEOPAX.get_Neoclassical_Fluxes = get_Neoclassical_Fluxes
    if not hasattr(NEOPAX, "Monoenergetic"):
        from NEOPAX._database import Monoenergetic

        NEOPAX.Monoenergetic = Monoenergetic
    return NEOPAX


def _has_local_boundary_stack() -> bool:
    return (
        find_neopax_root() is not None
        and find_vmec_jax_example_input() is not None
        and find_booz_xform_jax_root() is not None
    )


def _make_species(NEOPAX, field):
    rho = jnp.asarray(field.rho_grid)
    te = 1500.0 - 500.0 * rho**2
    ti = 1200.0 - 400.0 * rho**2
    ne = 2.0e19 - 0.5e19 * rho**2
    ni = ne
    return NEOPAX.Species(
        2,
        int(field.n_r),
        jnp.arange(2),
        jnp.asarray([1.0 / 1836.15267343, 1.0]),
        jnp.asarray([-1.0, 1.0]),
        jnp.stack([te, ti]),
        jnp.stack([ne, ni]),
        jnp.zeros_like(field.r_grid),
        field.r_grid,
        field.r_grid_half,
        field.dr,
        field.Vprime_half,
        field.overVprime,
        jnp.asarray([ne[-1], ni[-1]]),
        jnp.asarray([te[-1], ti[-1]]),
    )


def _synthetic_imported_surface(*, iota=0.4, b0=2.0, b10=0.2, b_theta=0.1, b_zeta=1.0):
    return BoozerSurface(
        m=jnp.asarray([0, 1, 2]),
        n=jnp.asarray([0, 0, 1]),
        b_cos=jnp.asarray([b0, b10, 0.03]),
        nfp=1,
        iota=jnp.asarray(iota),
        psi_p=jnp.asarray(1.2),
        b_theta=jnp.asarray(b_theta),
        b_zeta=jnp.asarray(b_zeta),
        b0=jnp.asarray(b0),
    )


def test_to_neopax_monoenergetic_preserves_jax_scalar_a_b():
    scan = ModuleType("scan")
    scan.rho = jnp.asarray([0.25, 0.5])
    scan.nu_v = jnp.asarray([1.0e-3, 1.0e-2])
    scan.Er = jnp.asarray([[0.0, 1.0e-4], [0.0, 2.0e-4]])
    scan.Es = jnp.asarray([[0.0, 1.0e-4], [0.0, 2.0e-4]])
    scan.drds = jnp.asarray([1.0, 1.0])
    scan.D11 = jnp.asarray(
        [
            [[1.0e-3, 2.0e-3], [3.0e-3, 4.0e-3]],
            [[1.1e-3, 2.1e-3], [3.1e-3, 4.1e-3]],
        ]
    )
    scan.D13 = jnp.asarray(
        [
            [[1.0e-2, 2.0e-2], [3.0e-2, 4.0e-2]],
            [[1.1e-2, 2.1e-2], [3.1e-2, 4.1e-2]],
        ]
    )
    scan.D33 = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.1, 2.1], [3.1, 4.1]],
        ]
    )
    scan.D33_spitzer = scan.D33 + 0.5

    fake = ModuleType("NEOPAX")
    fake.Monoenergetic = lambda **kwargs: type("FakeMonoenergetic", (), kwargs)()
    sys.modules["NEOPAX"] = fake

    a_b = jnp.asarray(1.5)
    mapped = to_neopax_monoenergetic(scan, a_b=a_b)
    assert not isinstance(mapped.a_b, float)
    assert jnp.allclose(jnp.asarray(mapped.a_b), a_b)
    sys.modules.pop("NEOPAX", None)


@pytest.mark.skipif(find_neopax_root() is None, reason="requires local NEOPAX checkout")
def test_differentiable_neopax_field_matches_external_constructor_except_b0_radius_bug():
    NEOPAX = _import_neopax()

    rho_half = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8])
    rho_full = jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    volume_p = jnp.asarray(80.0)
    vp = jnp.asarray([0.0, 12.0, 11.0, 10.0, 9.0])
    iotaf = jnp.asarray([0.0, 0.5, 0.45, 0.4, 0.35])
    psia = jnp.asarray(1.2)
    bmnc_b = jnp.asarray(
        [
            [5.0, 0.50],
            [5.2, 0.45],
            [5.4, 0.40],
            [5.6, 0.35],
        ]
    )
    rmnc_b = jnp.asarray(
        [
            [6.0, 0.0],
            [6.1, 0.0],
            [6.2, 0.0],
            [6.3, 0.0],
        ]
    )
    gmnc_b = jnp.asarray(
        [
            [1.4, 0.0],
            [1.3, 0.0],
            [1.2, 0.0],
            [1.1, 0.0],
        ]
    )
    xm_b = jnp.asarray([0, 1], dtype=jnp.int32)
    xn_b = jnp.asarray([0, 0], dtype=jnp.int32)
    bvco = jnp.asarray([0.0, 20.0, 19.0, 18.0, 17.0])
    buco = jnp.asarray([0.0, 0.5, 0.6, 0.7, 0.8])

    field = build_differentiable_neopax_field(
        n_r=5,
        rho_half=rho_half,
        rho_full=rho_full,
        volume_p=volume_p,
        vp=vp,
        iotaf=iotaf,
        Psia=psia,
        bmnc_b=bmnc_b,
        rmnc_b=rmnc_b,
        gmnc_b=gmnc_b,
        xm_b=xm_b,
        xn_b=xn_b,
        bvco=bvco,
        buco=buco,
    )
    reference = NEOPAX.Field(
        5,
        rho_half,
        rho_full,
        volume_p,
        vp,
        iotaf,
        psia,
        bmnc_b,
        rmnc_b,
        gmnc_b,
        xm_b,
        xn_b,
        bvco,
        buco,
    )

    assert jnp.allclose(field.rho_grid, reference.rho_grid)
    assert jnp.allclose(field.rho_grid_half, reference.rho_grid_half)
    assert jnp.allclose(field.r_grid, reference.r_grid)
    assert jnp.allclose(field.Vprime, reference.Vprime)
    assert jnp.allclose(field.Vprime_half, reference.Vprime_half)
    assert jnp.allclose(field.iota, reference.iota)
    assert jnp.allclose(field.B_10, reference.B_10, equal_nan=True)
    expected_b0 = 4.8 + field.rho_grid
    expected_bsqav = (field.G_value + field.iota * field.I_value) / (
        field.sqrtg00_value * expected_b0**2
    )
    assert not jnp.allclose(field.B0, reference.B0, equal_nan=True)
    assert jnp.allclose(field.B0, expected_b0)
    assert jnp.allclose(field.Bsqav, expected_bsqav, equal_nan=True)
    assert not jnp.allclose(field.Bsqav, reference.Bsqav, equal_nan=True)


def test_differentiable_neopax_field_without_b10_mode_is_axis_safe():
    field = build_differentiable_neopax_field(
        n_r=1,
        rho_half=jnp.asarray([0.0, 0.3, 0.6, 0.9]),
        rho_full=jnp.asarray([0.0, 0.25, 0.5, 0.75]),
        volume_p=jnp.asarray(10.0),
        vp=jnp.asarray([0.0, 1.0, 1.0, 1.0]),
        iotaf=jnp.asarray([0.0, 0.4, 0.3, 0.2]),
        Psia=jnp.asarray(1.0),
        bmnc_b=jnp.asarray([[2.0], [2.1], [2.2]]),
        rmnc_b=jnp.asarray([[5.0], [5.0], [5.0]]),
        gmnc_b=jnp.asarray([[1.0], [1.0], [1.0]]),
        xm_b=jnp.asarray([0], dtype=jnp.int32),
        xn_b=jnp.asarray([0], dtype=jnp.int32),
        bvco=jnp.asarray([0.0, 1.0, 1.0, 1.0]),
        buco=jnp.asarray([0.0, 0.1, 0.1, 0.1]),
        r0_override=jnp.asarray(5.0),
    )

    assert field.n_r == 1
    assert field.dr == 0.0
    assert jnp.allclose(field.B_10, jnp.zeros_like(field.B_10))
    assert jnp.all(jnp.isfinite(field.Bsqav))


def test_neopax_field_scalar_helpers_cover_zero_and_mode_branches():
    assert jnp.allclose(
        _safe_divide(jnp.asarray([2.0, 2.0]), jnp.asarray([1.0, 0.0])),
        jnp.asarray([2.0, 0.0]),
    )
    assert jnp.allclose(_safe_reciprocal(jnp.asarray([2.0, 0.0])), jnp.asarray([0.5, 0.0]))
    assert _find_mode_index(jnp.asarray([0, 1]), jnp.asarray([0, 0]), m_value=1, n_value=0) == 1
    assert _find_mode_index(jnp.asarray([0]), jnp.asarray([0]), m_value=1, n_value=0) is None
    assert jnp.allclose(_rho_half_mesh_from_s(jnp.asarray([0.25])), jnp.asarray([0.5]))
    assert jnp.allclose(
        _rho_half_mesh_from_s(jnp.asarray([0.0, 0.25, 1.0])),
        jnp.asarray([0.0, 0.35355338, 0.79056942]),
    )

    iota, b_theta, b_zeta, gmnc = _apply_boozer_sign_convention_profiles(
        iotaf=jnp.asarray([0.0, 0.4, -0.3]),
        buco=jnp.asarray([0.0, 0.2, 0.1]),
        bvco=jnp.asarray([0.0, 1.0, -0.1]),
        gmnc_b=jnp.asarray([[2.0, 0.1], [3.0, 0.2]]),
    )
    assert iota.shape == (3,)
    assert b_theta.shape == (3,)
    assert b_zeta.shape == (3,)
    assert gmnc.shape == (2, 2)


def test_surface_helpers_use_b10_fallbacks():
    with_b10 = BoozerSurface(
        m=jnp.asarray([0, 1]),
        n=jnp.asarray([0, 0]),
        b_cos=jnp.asarray([2.0, 0.4]),
        nfp=1,
        iota=0.4,
        psi_p=1.0,
        b_theta=0.1,
        b_zeta=1.0,
        b0=2.0,
    )
    without_b10 = BoozerSurface(
        m=jnp.asarray([0, 2]),
        n=jnp.asarray([0, 1]),
        b_cos=jnp.asarray([2.0, 0.1]),
        nfp=1,
        iota=0.4,
        psi_p=1.0,
        b_theta=0.1,
        b_zeta=1.0,
        b0=None,
    )

    assert jnp.allclose(_surface_b10(with_b10), 0.2)
    assert jnp.allclose(_surface_b10(without_b10), 0.0)
    assert jnp.isfinite(_surface_bsqav(with_b10, ntheta=5, nzeta=5))


def test_vmec_scalar_profile_helpers_with_fake_modules(monkeypatch):
    integrals = ModuleType("vmec_jax.integrals")
    integrals.cumrect_s_halfmesh = lambda values, s: jnp.cumsum(jnp.asarray(values))
    energy = ModuleType("vmec_jax.energy")
    energy.flux_profiles_from_indata = lambda indata, s, signgs: SimpleNamespace(
        phipf=jnp.asarray([0.0, -0.2, -0.3])
    )
    forces = ModuleType("vmec_jax.vmec_forces")
    forces.vmec_forces_rz_from_wout = lambda **kwargs: SimpleNamespace(bc="bc")
    residue = ModuleType("vmec_jax.vmec_residue")
    residue.vmec_force_norms_from_bcovar_dynamic = lambda **kwargs: SimpleNamespace(
        volume=jnp.asarray(-2.0),
        vp=jnp.asarray([0.0, -1.0, -2.0]),
    )
    monkeypatch.setitem(sys.modules, "vmec_jax.integrals", integrals)
    monkeypatch.setitem(sys.modules, "vmec_jax.energy", energy)
    monkeypatch.setitem(sys.modules, "vmec_jax.vmec_forces", forces)
    monkeypatch.setitem(sys.modules, "vmec_jax.vmec_residue", residue)

    static = SimpleNamespace(
        s=jnp.asarray([0.0, 0.5, 1.0]),
        cfg=SimpleNamespace(nfp=5, mpol=3, ntor=2, lasym=False),
        trig_vmec="trig",
    )
    state = SimpleNamespace(
        phipf_out=jnp.asarray([0.0, -0.2, -0.3]),
        Rcos=jnp.asarray([[0.0], [3.0]]),
    )

    assert jnp.allclose(_vmec_psia_from_state(state, static), 0.5)
    assert jnp.allclose(_vmec_psia_from_indata(indata=object(), static=static, signgs=-1), 0.5)
    assert jnp.allclose(_vmec_edge_r00_from_state(state), 3.0)
    volume, vp = _vmec_volume_profiles_from_state(
        state=state,
        static=static,
        indata=object(),
        signgs=-1,
    )
    assert jnp.allclose(volume, 8.0 * jnp.pi**2)
    assert jnp.allclose(vp, jnp.asarray([0.0, 1.0, 2.0]))

    with pytest.raises(AttributeError, match="phipf_out"):
        _vmec_psia_from_state(SimpleNamespace(phipf_out=None), static)
    with pytest.raises(AttributeError, match="Rcos"):
        _vmec_edge_r00_from_state(SimpleNamespace(Rcos=None))


def test_differentiable_neopax_field_from_vmec_state_uses_axis_safe_profiles(monkeypatch):
    import ntx._neopax_vmec_jax_field as neopax_field_module

    static = SimpleNamespace(s=jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0]))
    state = SimpleNamespace(
        phipf_out=jnp.asarray([0.0, -0.2, -0.3]),
        Rcos=jnp.asarray([[0.0], [6.0]]),
    )
    calls: dict[str, object] = {}

    def fake_surfaces_from_state(**kwargs):
        calls["s_values"] = kwargs["s_values"]
        return (
            _synthetic_imported_surface(iota=0.35, b0=2.1, b10=0.21, b_theta=0.12, b_zeta=1.05),
            _synthetic_imported_surface(iota=0.45, b0=2.2, b10=0.11, b_theta=0.15, b_zeta=1.10),
        )

    monkeypatch.setattr(
        neopax_field_module,
        "_vmec_volume_profiles_from_state",
        lambda **kwargs: (
            jnp.asarray(80.0),
            jnp.asarray([0.0, 12.0, 11.0, 10.0, 9.0]),
        ),
    )
    monkeypatch.setattr(
        neopax_field_module,
        "_vmec_psia_from_state",
        lambda state, static: jnp.asarray(1.2),
    )
    monkeypatch.setattr(
        neopax_field_module,
        "_vmec_edge_r00_from_state",
        lambda state: jnp.asarray(6.0),
    )
    monkeypatch.setattr(
        neopax_field_module,
        "surfaces_from_vmec_jax_state",
        fake_surfaces_from_state,
    )

    field = build_differentiable_neopax_field_from_vmec_jax_state(
        state=state,
        static=static,
        indata=object(),
        signgs=-1,
        n_r=4,
        mboz=3,
        nboz=3,
    )

    assert calls["s_values"] == pytest.approx((1.0 / 9.0, 4.0 / 9.0))
    assert field.n_r == 4
    assert jnp.allclose(field.r_grid[0], 0.5 * field.r_grid[1])
    assert jnp.allclose(field.overVprime[0], 0.0)
    assert jnp.allclose(field.B_10[0], 0.0)
    assert jnp.allclose(field.B_10[-1], field.B_10[-2])
    assert jnp.allclose(field.iota[0], 0.0)
    assert jnp.allclose(field.iota[-1], field.iota[-2])
    assert jnp.all(jnp.isfinite(field.Bsqav))
    assert jnp.all(jnp.isfinite(field.sqrtg00_value))


def test_differentiable_neopax_field_from_vmec_state_falls_back_to_indata_psia(monkeypatch):
    import ntx._neopax_vmec_jax_field as neopax_field_module

    monkeypatch.setattr(
        neopax_field_module,
        "_vmec_volume_profiles_from_state",
        lambda **kwargs: (
            jnp.asarray(40.0),
            jnp.asarray([0.0, 5.0, 4.0]),
        ),
    )
    monkeypatch.setattr(
        neopax_field_module,
        "_vmec_psia_from_indata",
        lambda **kwargs: jnp.asarray(0.75),
    )
    monkeypatch.setattr(
        neopax_field_module,
        "_vmec_edge_r00_from_state",
        lambda state: jnp.asarray(5.0),
    )
    monkeypatch.setattr(
        neopax_field_module,
        "surfaces_from_vmec_jax_state",
        lambda **kwargs: (_synthetic_imported_surface(),),
    )

    field = build_differentiable_neopax_field_from_vmec_jax_state(
        state=SimpleNamespace(Rcos=jnp.asarray([[0.0], [5.0]])),
        static=SimpleNamespace(s=jnp.asarray([0.0, 0.5, 1.0])),
        indata=object(),
        signgs=1,
        n_r=3,
    )

    assert jnp.allclose(field.Psia_value, 0.75)
    assert field.r_grid.shape == (3,)
    assert jnp.all(jnp.isfinite(field.G_PS))


def test_differentiable_neopax_fluxes_copy_axis_block_and_apply_lij_forces(monkeypatch):
    fake_neoclassical = ModuleType("NEOPAX._neoclassical")

    def fake_lij_matrix(species, grid, field, database, species_index, radial_index):
        base = 100.0 * species_index + 10.0 * radial_index
        return jnp.asarray(
            [
                [base + 1.0, base + 2.0, base + 3.0],
                [base + 4.0, base + 5.0, base + 6.0],
                [base + 7.0, base + 8.0, base + 9.0],
            ]
        )

    fake_neoclassical.get_Lij_matrix = fake_lij_matrix
    monkeypatch.setitem(sys.modules, "NEOPAX", ModuleType("NEOPAX"))
    monkeypatch.setitem(sys.modules, "NEOPAX._neoclassical", fake_neoclassical)

    species = SimpleNamespace(
        species_indeces=jnp.asarray([0, 1]),
        A1=jnp.asarray([[1.0, 1.1, 1.2], [0.7, 0.8, 0.9]]),
        A2=jnp.asarray([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]),
        A3=jnp.asarray([0.05, 0.06, 0.07]),
        temperature=jnp.asarray([[2.0, 2.1, 2.2], [3.0, 3.1, 3.2]]),
        density=jnp.asarray([[4.0, 4.1, 4.2], [5.0, 5.1, 5.2]]),
    )
    grid = SimpleNamespace(full_grid_indeces=jnp.asarray([0, 1, 2]))
    lij, gamma, heat, upar = get_differentiable_neopax_fluxes(
        species,
        grid,
        field=object(),
        database=object(),
    )

    assert lij.shape == (2, 3, 3, 3)
    assert jnp.allclose(lij[:, 0], lij[:, 1])
    expected_gamma_axis = -species.density[0, 0] * (
        lij[0, 0, 0, 0] * species.A1[0, 0]
        + lij[0, 0, 0, 1] * species.A2[0, 0]
        + lij[0, 0, 0, 2] * species.A3[0]
    )
    expected_heat_axis = (
        -species.temperature[1, 0]
        * species.density[1, 0]
        * (
            lij[1, 0, 1, 0] * species.A1[1, 0]
            + lij[1, 0, 1, 1] * species.A2[1, 0]
            + lij[1, 0, 1, 2] * species.A3[0]
        )
    )
    expected_upar_edge = -species.density[1, 2] * (
        lij[1, 2, 2, 0] * species.A1[1, 2]
        + lij[1, 2, 2, 1] * species.A2[1, 2]
        + lij[1, 2, 2, 2] * species.A3[2]
    )
    assert jnp.allclose(gamma[0, 0], expected_gamma_axis)
    assert jnp.allclose(heat[1, 0], expected_heat_axis)
    assert jnp.allclose(upar[1, 2], expected_upar_edge)


def test_boundary_params_field_builder_delegates_to_state_builder(monkeypatch):
    import ntx._neopax_vmec_jax_field as neopax_field_module

    context = SimpleNamespace(static="static", indata="indata", signgs=-1)
    calls = {}

    def fake_solve(ctx, params, **kwargs):
        calls["solve"] = (ctx, params, kwargs)
        return "state"

    def fake_build(**kwargs):
        calls["build"] = kwargs
        return "field"

    monkeypatch.setattr(neopax_field_module, "solve_vmec_jax_boundary_state", fake_solve)
    monkeypatch.setattr(
        neopax_field_module,
        "build_differentiable_neopax_field_from_vmec_jax_state",
        fake_build,
    )

    result = build_differentiable_neopax_field_from_vmec_jax_boundary_params(
        context,
        jnp.asarray([0.0]),
        n_r=7,
        vmec_project=False,
        max_iter=3,
        step_size=0.25,
        ftol=1.0e-9,
        implicit="implicit",
        mboz=4,
        nboz=5,
        apply_boozer_sign_convention=False,
    )

    assert result == "field"
    assert calls["solve"][2]["max_iter"] == 3
    assert calls["build"]["state"] == "state"
    assert calls["build"]["static"] == "static"
    assert calls["build"]["apply_boozer_sign_convention"] is False


def test_booz_xform_gmnc_helpers_with_fake_internal_api(monkeypatch):
    import ntx._neopax_vmec_jax_boozer as vmec_jax_boozer_module

    jax_api = ModuleType("booz_xform_jax.jax_api")

    def prepare_booz_xform_constants_from_inputs(**kwargs):
        return (
            SimpleNamespace(
                mmax_non=1,
                nmax_non=1,
                mmax_nyq=1,
                nmax_nyq=1,
                nfp=1,
                nzeta=2,
                nu2_b=2,
            ),
            SimpleNamespace(
                theta_grid=jnp.asarray([0.0, 1.0]),
                zeta_grid=jnp.asarray([0.0, 1.0]),
                xm_b=jnp.asarray([0, 1]),
                xn_b=jnp.asarray([0, -1]),
            ),
        )

    def init_trig(theta_grid, zeta_grid, mmax, nmax, nfp):
        size = theta_grid.shape[0]
        width = max(int(mmax), int(nmax)) + 1
        base = jnp.ones((size, width))
        return base, jnp.zeros_like(base), base, jnp.zeros_like(base)

    def surface_transform(*args, **kwargs):
        bmnc = args[3]
        bsubvmnc = args[5]
        return (bmnc, bmnc, bmnc, bmnc, bmnc + bsubvmnc)

    jax_api.prepare_booz_xform_constants_from_inputs = prepare_booz_xform_constants_from_inputs
    jax_api._init_trig = init_trig
    jax_api._surface_transform = surface_transform
    monkeypatch.setitem(sys.modules, "booz_xform_jax.jax_api", jax_api)

    inputs = SimpleNamespace(
        xm=jnp.asarray([0, 1]),
        xn=jnp.asarray([0, -1]),
        xm_nyq=jnp.asarray([0, 1]),
        xn_nyq=jnp.asarray([0, -1]),
        rmnc=jnp.asarray([[1.0, 0.1]]),
        zmns=jnp.asarray([[0.0, 0.1]]),
        lmns=jnp.asarray([[0.0, 0.1]]),
        bmnc=jnp.asarray([[2.0, 0.2]]),
        bsubumnc=jnp.asarray([[0.1, 0.2]]),
        bsubvmnc=jnp.asarray([[0.3, 0.4]]),
        iota=jnp.asarray([0.5]),
        bmns=None,
        bsubumns=None,
        bsubvmns=None,
    )
    gmnc = _booz_xform_gmnc_from_inputs(inputs=inputs, mboz=2, nboz=2, asym=False)
    assert jnp.allclose(gmnc, jnp.asarray([[2.3, 0.6]]))

    monkeypatch.setattr(
        vmec_jax_boozer_module,
        "_booz_xform_bundle_from_vmec_jax_state",
        lambda **kwargs: (inputs, {"bmnc_b": jnp.asarray([[2.0, 0.2]])}),
    )
    bundle_inputs, out = _booz_xform_bundle_with_gmnc_from_vmec_jax_state(
        state="state",
        static=SimpleNamespace(cfg=SimpleNamespace(lasym=False)),
        indata="indata",
        signgs=1,
        mboz=2,
        nboz=2,
    )
    assert bundle_inputs is inputs
    assert "gmnc_b" in out

    empty_api = ModuleType("booz_xform_jax.jax_api")
    empty_api.prepare_booz_xform_constants_from_inputs = prepare_booz_xform_constants_from_inputs
    monkeypatch.setitem(sys.modules, "booz_xform_jax.jax_api", empty_api)
    with pytest.raises(RuntimeError, match="internal JAX helpers"):
        _booz_xform_gmnc_from_inputs(inputs=inputs, mboz=2, nboz=2, asym=False)


@pytest.mark.skipif(
    not _has_local_boundary_stack() or os.environ.get("NTX_RUN_BOUNDARY_AUTODIFF") != "1",
    reason=(
        "requires local vmec_jax, booz_xform_jax, and NEOPAX checkouts plus "
        "NTX_RUN_BOUNDARY_AUTODIFF=1; the full reverse-mode compile is memory intensive"
    ),
)
def test_boundary_to_neopax_current_objective_is_differentiable():
    NEOPAX = _import_neopax()

    context = build_vmec_jax_boundary_context(
        find_vmec_jax_example_input(),
        max_mode=1,
        include=("rc", "zs"),
        fix=("rc00",),
    )
    if len(context.specs) == 0:
        pytest.skip("vmec_jax example did not expose any boundary parameters")

    rho = jnp.asarray([0.25, 0.45, 0.65, 0.85])
    nu_v = jnp.logspace(-4, -2, 4)
    er_row = jnp.asarray([0.0, 1.0e-4, 3.0e-4, 1.0e-3])
    er = jnp.tile(er_row[None, :], (rho.shape[0], 1))
    grid = GridSpec(5, 5, 4)

    def objective(params):
        state = solve_vmec_jax_boundary_state(context, params, max_iter=3)
        field = build_differentiable_neopax_field_from_vmec_jax_state(
            state=state,
            static=context.static,
            indata=context.indata,
            signgs=context.signgs,
            n_r=11,
            mboz=12,
            nboz=12,
        )
        drds = field.a_b * 0.5 / jnp.clip(rho, 0.05, None)
        scan = build_ntx_neopax_scan_from_vmec_jax_state(
            state=state,
            static=context.static,
            indata=context.indata,
            signgs=context.signgs,
            rho=rho,
            nu_v=nu_v,
            Er=er,
            Es=er,
            drds=drds,
            grid=grid,
            psi_p=field.Psia_value,
            source_name="vmec_jax_boundary_autodiff_smoke",
        )
        database = to_neopax_monoenergetic(scan, a_b=field.a_b)
        species = _make_species(NEOPAX, field)
        neopax_grid = NEOPAX.Grid.create_standard(int(field.n_r), 12, 2)
        _, _, _, upar = get_differentiable_neopax_fluxes(species, neopax_grid, field, database)
        total_current = jnp.sum(species.charge[:, None] * upar * field.Vprime[None, :] * field.dr)
        return total_current

    params0 = jnp.zeros((len(context.specs),), dtype=jnp.float64)
    value = objective(params0)
    gradient = jax.grad(objective)(params0)

    assert jnp.isfinite(value)
    assert gradient.shape == params0.shape
    assert jnp.all(jnp.isfinite(gradient))
