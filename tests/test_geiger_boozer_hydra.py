"""Geiger/IPP Boozer-coordinate reader and the NTX against DKES comparison on hydra_Np04.

The DKES results and the Geiger Boozer file of the HELIAS reactor hydra_Np04_20190108 cannot be
redistributed. The reader is tested on a small synthetic file written here; the comparison with
DKES reads the private data from ``NTX_DKES_HYDRA_DIR`` and ``NTX_HYDRA_BOOZER`` and skips without.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

import ntx

ROOT = Path(__file__).resolve().parents[1]
DKES_DIR = os.environ.get("NTX_DKES_HYDRA_DIR")
BOOZER = os.environ.get("NTX_HYDRA_BOOZER")
needs_private = pytest.mark.skipif(
    not (DKES_DIR and BOOZER and Path(DKES_DIR).is_dir() and Path(BOOZER).is_file()),
    reason="NTX_DKES_HYDRA_DIR and NTX_HYDRA_BOOZER (private hydra_Np04 data) not set",
)
MU0 = 4.0e-7 * math.pi
NPER = 4

_MODES = [(0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]


def _surface_block(s: float, iota: float, curr_pol: float, curr_tor: float, scale: float) -> str:
    lines = [
        "       s         iota  curr_pol/nper    curr_tor    pprime   sqrt g(0,0)",
        "                            [A]            [A]   dp/ds,[Pa] (dV/ds)/nper",
        f"  {s:.4E}  {iota:.4E} {curr_pol:.4E}  {curr_tor:.4E} -1.0000E+04 -4.0000E+02",
        "    m    n        r/[m]           z/[m] (phib-phi)*nper/twopi     bmn/[T]",
    ]
    for k, (m, n) in enumerate(_MODES):
        bmn = 5.0 * scale if (m, n) == (0, 0) else 0.1 * scale * (k + 1)
        lines.append(f"{m:5d}{n:5d}  1.0E+00  2.0E-01  3.0E-02  {bmn:.8E}")
    return "\n".join(lines)


def _write_file(path: Path) -> Path:
    header = "\n".join(
        [
            "CC Boozer-coordinate data file",
            "CC Version 0.1",
            "CC Configurationname = ",
            " m0b  n0b nsurf nper  flux/[Tm^2]     a/[m]     R/[m]   avol/[m]  Rvol/[m] Vol/[m^3]",
            f"    1    1    3    {NPER} -6.120000E+01   2.01879  23.61794   2.02211  23.54054 "
            "1900.0072   original: a=1.84957 m",
        ]
    )
    blocks = [
        _surface_block(0.1, 0.90, -1.0e8, 20.0, 1.00),
        _surface_block(0.2, 0.94, -1.2e8, 30.0, 1.02),
        _surface_block(0.4, 1.00, -1.4e8, 40.0, 1.04),
    ]
    path.write_text(header + "\n" + "\n".join(blocks) + "\n")
    return path


def test_reader_converts_a_surface_and_accepts_the_ten_number_header(tmp_path: Path) -> None:
    surface = ntx.load_geiger_boozer_surface(
        _write_file(tmp_path / "boozer.data"), rho=math.sqrt(0.2)
    )
    assert surface.nfp == NPER
    assert surface.iota == pytest.approx(-0.94)
    assert float(surface.b_zeta) == pytest.approx(1.2e8 * NPER / (2 * math.pi) * MU0)
    assert float(surface.b_theta) == pytest.approx(30.0 / (2 * math.pi) * MU0)
    assert float(surface.b0) == pytest.approx(5.1)
    psi = 0.2 * 61.2 / (2 * math.pi)
    assert float(surface.psi_p) == pytest.approx(
        float(surface.b0) * math.sqrt(2 * psi / float(surface.b0))
    )
    assert list(np.asarray(surface.m)) == [m for m, _ in _MODES]
    assert list(np.asarray(surface.n)) == [n for _, n in _MODES]


def test_reader_interpolates_linearly_in_s_and_clamps(tmp_path: Path) -> None:
    path = _write_file(tmp_path / "boozer.data")
    mid = ntx.load_geiger_boozer_surface(path, s=0.15)
    assert mid.iota == pytest.approx(-0.92)
    assert float(mid.b0) == pytest.approx(5.0 * 1.01)
    assert float(mid.b_zeta) == pytest.approx(1.1e8 * NPER / (2 * math.pi) * MU0)
    edge = ntx.load_geiger_boozer_surface(path, s=0.9)
    assert edge.iota == pytest.approx(-1.00)
    assert float(ntx.load_geiger_boozer_surface(path, s=0.0).b0) == pytest.approx(5.0)
    trimmed = ntx.load_geiger_boozer_surface(path, s=0.15, min_bmn=0.05)
    assert len(trimmed.m) < len(mid.m)
    assert (int(trimmed.m[0]), int(trimmed.n[0])) == (0, 0)


def test_reader_rejects_bad_input(tmp_path: Path) -> None:
    path = _write_file(tmp_path / "boozer.data")
    with pytest.raises(ValueError):
        ntx.load_geiger_boozer_surface(path)
    with pytest.raises(ValueError):
        ntx.load_geiger_boozer_surface(path, rho=0.3, s=0.1)
    bad = tmp_path / "bad.data"
    bad.write_text("CC nothing here\n")
    with pytest.raises(ValueError):
        ntx.load_geiger_boozer_surface(bad, rho=0.3)


def test_surface_solves_in_the_collisional_limit(tmp_path: Path) -> None:
    surface = ntx.load_geiger_boozer_surface(
        _write_file(tmp_path / "boozer.data"), rho=math.sqrt(0.2)
    )
    prepared = ntx.prepare_monoenergetic_system(surface, ntx.GridSpec(9, 9, 16))
    result = ntx.solve_prepared(prepared, ntx.MonoenergeticCase(nu_hat=10.0, epsi_hat=0.0))
    assert float(result.D33) / float(result.D33_spitzer) == pytest.approx(1.0, abs=1e-3)


def _script():
    spec = importlib.util.spec_from_file_location(
        "validate_dkes_hydra_np04", ROOT / "scripts" / "validate_dkes_hydra_np04.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_record_holds_statistics_only() -> None:
    record = json.loads((ROOT / "benchmarks" / "results" / "dkes_hydra_np04.json").read_text())
    text = json.dumps(record["agreement"])
    assert all(token not in text for token in ("D11_S", "D31_S", "D33_S", "Erstar", "nuprime"))
    assert {k for row in record["agreement"] for k in row} == {
        "rho",
        "efield",
        "regime",
        "points",
        "d11_median_pct",
        "d11_max_pct",
        "d33_max_pct",
        "d31_max_abs",
    }
    assert not _script().gate_errors(record["agreement"], {("0.875", "zero")})


@needs_private
def test_collisional_limit_of_dkes_and_ntx_is_d33_star_one() -> None:
    module = _script()
    rows = [r for r in module.reference_rows(DKES_DIR) if r["cmul"] == 10.0 and r["efield"] == 0.0]
    assert len(rows) == 6  # rho = 0.5 has no efield = 0 point
    for row in rows:
        (point,) = module.solve_columns(Path(BOOZER), row["rho"], [row], grid=(9, 17, 24))
        assert point["dkes"]["D33_star"] == pytest.approx(1.0, abs=2e-3)
        assert point["ntx"]["D33_star"] == pytest.approx(1.0, abs=2e-3)


@needs_private
def test_ntx_agrees_with_dkes_at_the_inner_radius() -> None:
    module = _script()
    points = module.compare(Path(BOOZER), Path(DKES_DIR), rhos=["0.25"])
    stats = module.agreement(points)
    assert stats == module.agreement(points)
    assert not module.gate_errors(stats)
    record = json.loads((ROOT / "benchmarks" / "results" / "dkes_hydra_np04.json").read_text())
    assert [s for s in record["agreement"] if s["rho"] == "0.25"] == stats


def test_harmonics_beyond_the_grid_nyquist_are_dropped_not_aliased() -> None:
    import jax.numpy as jnp

    from ntx.geometry import geometry_on_grid

    def surface(m, n, b):
        return ntx.BoozerSurface(
            m=jnp.asarray(m),
            n=jnp.asarray(n),
            b_cos=jnp.asarray(b),
            nfp=4,
            iota=0.9,
            psi_p=1.0,
            b_theta=0.0,
            b_zeta=20.0,
            b0=5.0,
        )

    spec = ntx.GridSpec(9, 9, 8)  # represents m <= 4 and |n| <= 4
    full = surface([0, 1, 4, 7, 1], [0, 0, 4, 1, 6], [5.0, 0.2, 0.01, 0.05, 0.05])
    kept = surface([0, 1, 4], [0, 0, 4], [5.0, 0.2, 0.01])
    got = geometry_on_grid(full, spec)
    ref = geometry_on_grid(kept, spec)
    np.testing.assert_allclose(np.asarray(got.b), np.asarray(ref.b), rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(got.radial_drift_spatial),
        np.asarray(ref.radial_drift_spatial),
        rtol=0,
        atol=1e-14,
    )
    # A finer grid represents the m = 7 harmonic again.
    assert not np.allclose(
        np.asarray(geometry_on_grid(full, ntx.GridSpec(17, 17, 8)).b[:1, :1]), 5.2
    )
