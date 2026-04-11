from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

SAMPLE_DKES = FIXTURES / "sample_surface.ddkes2.data"
SAMPLE_MAGNETIC = FIXTURES / "sample_magnetic_configuration.dat"
SAMPLE_WOUT = FIXTURES / "sample_wout.nc"
SAMPLE_BOOZMN = FIXTURES / "sample_boozmn.nc"
SAMPLE_NEOPAX = FIXTURES / "sample_neopax_scan.h5"
