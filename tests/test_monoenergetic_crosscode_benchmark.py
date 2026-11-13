"""The committed cross-code benchmark record backs the numbers quoted in the docs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "bench_monoenergetic_crosscode.py"
# LHD at nu* = 3e-4 is not resolved: D31 still moves 1.9% between the two finest
# grids, (39, 87, 280) and (47, 103, 360). The docs report it as such.
UNRESOLVED = {("LHD", "1/nu")}
RECORD = json.loads((ROOT / "benchmarks" / "results" / "monoenergetic_crosscode.json").read_text())


def _load_script():
    spec = importlib.util.spec_from_file_location("bench_monoenergetic_crosscode", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_device_and_regime_was_measured_against_a_checked_reference():
    bench = _load_script()
    seen = {(row["device"], row["regime"]) for row in RECORD["rows"]}
    assert seen == {(d, r) for d in bench.SOURCES for r in bench.REGIMES}
    for row in RECORD["rows"]:
        if (row["device"], row["regime"]) in UNRESOLVED:
            assert max(row["reference_check"]) > 1e-2
            continue
        assert max(row["reference_check"]) < 5e-3, (row["device"], row["regime"])


def test_ntx_and_monkes_agree_to_rounding_on_the_same_grid():
    for row in RECORD["rows"]:
        runs = {(r["code"], r["device"]): r for r in row["runs"]}
        ntx, monkes = runs["ntx", "cpu"]["values"], runs["monkes", "cpu"]["values"]
        np.testing.assert_allclose(
            ntx, monkes, rtol=1e-8, err_msg=f"{row['device']} {row['regime']}"
        )


def test_every_code_is_within_one_percent_of_the_reference():
    for row in RECORD["rows"]:
        if (row["device"], row["regime"]) in UNRESOLVED:
            continue
        for run in row["runs"]:
            assert max(run["rel_err"]) < 1e-2, (row["device"], row["regime"], run["code"])


def test_the_report_renders_from_the_record():
    table = _load_script().report()
    assert table.count("| W7-X |") >= 3
    assert "N_xi for 1%" in table
