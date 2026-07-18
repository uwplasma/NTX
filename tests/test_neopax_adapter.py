from __future__ import annotations

import sys
from types import ModuleType

import jax.numpy as jnp

from ntx import (
    GridSpec,
    build_ntx_neopax_scan,
    load_neopax_reference_scan,
    surface_from_vmex_vmec_wout_file,
    to_neopax_monoenergetic,
)
from ntx._checkout_paths import find_neopax_root

from .fixture_data import SAMPLE_NEOPAX, SAMPLE_WOUT


def test_reference_scan_round_trips_into_neopax_constructor():
    scan = load_neopax_reference_scan(SAMPLE_NEOPAX)
    fake = ModuleType("NEOPAX")
    fake.Monoenergetic = lambda **kwargs: type("FakeMonoenergetic", (), kwargs)()
    sys.modules["NEOPAX"] = fake
    mapped = to_neopax_monoenergetic(scan, a_b=1.5)
    assert mapped.rho.shape == scan.rho.shape
    assert mapped.nu_log.shape == scan.nu_v.shape
    assert mapped.Er_list.shape == scan.Er.shape
    assert mapped.D11_log.shape == scan.D11.shape
    assert mapped.D13.shape == scan.D13.shape
    assert mapped.D33.shape == scan.D33.shape
    assert jnp.all(jnp.isfinite(mapped.D11_log))
    assert jnp.all(jnp.isfinite(mapped.D13))
    assert jnp.all(jnp.isfinite(mapped.D33))


def test_ntx_scan_maps_into_neopax_shapes():
    reference = load_neopax_reference_scan(SAMPLE_NEOPAX)
    rho = reference.rho
    nu_v = reference.nu_v
    Er = reference.Er
    Es = reference.Es
    drds = reference.drds

    def surface_loader(rho_value: float):
        return surface_from_vmex_vmec_wout_file(
            SAMPLE_WOUT,
            s=float(rho_value**2),
        )

    scan = build_ntx_neopax_scan(
        surface_loader,
        rho=rho,
        nu_v=nu_v,
        Es=Es,
        Er=Er,
        drds=drds,
        grid=GridSpec(n_theta=9, n_zeta=9, n_xi=8),
        source_name="sample_vmec_subset",
    )
    fake = ModuleType("NEOPAX")
    fake.Monoenergetic = lambda **kwargs: type("FakeMonoenergetic", (), kwargs)()
    sys.modules["NEOPAX"] = fake
    database = to_neopax_monoenergetic(scan, a_b=1.5)
    assert database.D11_log.shape == reference.D11.shape
    assert database.D13.shape == reference.D13.shape
    assert database.D33.shape == reference.D33.shape
    assert jnp.all(jnp.isfinite(database.D11_log))


def test_legacy_external_scan_round_trips_historical_d13_convention():
    neopax_root = find_neopax_root()
    if neopax_root is None:
        return

    sys.modules.pop("NEOPAX", None)
    try:
        import NEOPAX
        from NEOPAX._database import Monoenergetic
    except ImportError:  # pragma: no cover - local-only dependency
        return
    NEOPAX.Monoenergetic = Monoenergetic

    legacy_path = neopax_root / "tests" / "inputs" / "Dij_NEOPAX_FULL_S_NEW_W7X.h5"
    if not legacy_path.exists():
        return

    scan = load_neopax_reference_scan(legacy_path)
    mapped = to_neopax_monoenergetic(scan, a_b=5.5, d33_mode="raw")
    legacy = NEOPAX.Monoenergetic.read_monkes(5.5, str(legacy_path))

    assert jnp.allclose(mapped.D11_log, legacy.D11_log)
    assert jnp.allclose(mapped.D13, legacy.D13)
    assert jnp.allclose(mapped.D33, legacy.D33)


def test_generated_w7x_point_maps_into_database_convention():
    neopax_root = find_neopax_root()
    if neopax_root is None:
        return

    sys.modules.pop("NEOPAX", None)
    try:
        import NEOPAX
        from NEOPAX._database import Monoenergetic
    except ImportError:  # pragma: no cover - local-only dependency
        return
    NEOPAX.Monoenergetic = Monoenergetic

    legacy_path = neopax_root / "tests" / "inputs" / "Dij_NEOPAX_FULL_S_NEW_W7X.h5"
    wout_path = neopax_root / "tests" / "inputs" / "wout_W7-X_standard_configuration.nc"
    if not legacy_path.exists() or not wout_path.exists():
        return

    scan = load_neopax_reference_scan(legacy_path)
    # This point previously exposed the rebuilt-database D13 mapping bug in the
    # integrated W7-X workflow.
    rho_idx, nu_idx, er_idx = 6, 3, 9
    rho = float(scan.rho[rho_idx])
    nu_v = float(scan.nu_v[nu_idx])
    Er = float(scan.Er[rho_idx, er_idx])
    Es = float(scan.Es[rho_idx, er_idx])
    drds = float(scan.drds[rho_idx])

    generated = build_ntx_neopax_scan(
        lambda _: surface_from_vmex_vmec_wout_file(wout_path, s=float(rho**2)),
        rho=jnp.asarray([rho]),
        nu_v=jnp.asarray([nu_v]),
        Er=jnp.asarray([[Er]]),
        Es=jnp.asarray([[Es]]),
        drds=jnp.asarray([drds]),
        grid=GridSpec(n_theta=25, n_zeta=25, n_xi=63),
        source_name="w7x-one-point-regression",
    )
    mapped = to_neopax_monoenergetic(generated, a_b=5.5, d33_mode="raw")
    legacy = NEOPAX.Monoenergetic.read_monkes(5.5, str(legacy_path))

    assert jnp.allclose(mapped.D11_log[0, 0, 0], legacy.D11_log[rho_idx, nu_idx, er_idx])
    assert jnp.allclose(mapped.D13[0, 0, 0], legacy.D13[rho_idx, nu_idx, er_idx])
    assert jnp.allclose(mapped.D33[0, 0, 0], legacy.D33[rho_idx, nu_idx, er_idx])
    assert jnp.allclose(mapped.Er_list[0, 0], legacy.Er_list[rho_idx, er_idx])
