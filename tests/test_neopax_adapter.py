from __future__ import annotations

import sys
from types import ModuleType

import jax.numpy as jnp

from ntx import (
    GridSpec,
    build_ntx_neopax_scan,
    load_neopax_reference_scan,
    surface_from_vmec_jax_vmec_wout_file,
    to_neopax_monoenergetic,
)

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
        return surface_from_vmec_jax_vmec_wout_file(
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
