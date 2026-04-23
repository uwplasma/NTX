from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_coverage_report_writes_module_summary(tmp_path):
    coverage_json = tmp_path / "coverage.json"
    coverage_json.write_text(
        json.dumps(
            {
                "totals": {"percent_covered": 92.5},
                "files": {
                    "src/ntx/__main__.py": {
                        "summary": {
                            "percent_covered": 91.0,
                            "covered_lines": 91,
                            "missing_lines": 9,
                            "num_statements": 100,
                        }
                    },
                    str(ROOT / "src" / "ntx" / "solver.py"): {
                        "summary": {
                            "percent_covered": 88.0,
                            "covered_lines": 880,
                            "missing_lines": 120,
                            "num_statements": 1000,
                        }
                    },
                    str(ROOT / "src" / "ntx" / "grids.py"): {
                        "summary": {
                            "percent_covered": 97.5,
                            "covered_lines": 39,
                            "missing_lines": 1,
                            "num_statements": 40,
                        }
                    },
                    str(ROOT / "tests" / "test_solver.py"): {
                        "summary": {
                            "percent_covered": 100.0,
                            "covered_lines": 20,
                            "missing_lines": 0,
                            "num_statements": 20,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    json_output = tmp_path / "coverage-report.json"
    text_output = tmp_path / "coverage-report.txt"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_coverage_report.py"),
            "--json-input",
            str(coverage_json),
            "--json-output",
            str(json_output),
            "--text-output",
            str(text_output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(json_output.read_text(encoding="utf-8"))
    text = text_output.read_text(encoding="utf-8")

    assert report["overall_percent_covered"] == 92.5
    assert report["module_count"] == 3
    assert report["modules"][0]["module"] == "solver"
    assert report["modules"][1]["module"] == "__main__"
    assert report["modules"][2]["module"] == "grids"
    assert "Overall coverage: 92.5%" in text
    assert "- solver: 88.0% (880/1000 lines)" in text
