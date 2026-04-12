#!/usr/bin/env python3
"""Regenerate the manuscript-ready NTX figure bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def _manifest_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "_static",
        help="Directory for the generated figure bundle.",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default="inverse,profiles,science,validation,performance_smoke,performance_heavy",
        help="Comma-separated subset of figures to generate.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {item.strip() for item in args.figures.split(",") if item.strip()}
    manifest: dict[str, list[str]] = {}

    if "inverse" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "autodiff_inverse_problem.py"),
                "--output-prefix",
                str(output_dir / "autodiff_inverse_problem"),
            ]
        )
        manifest["inverse"] = [
            _manifest_path(output_dir / "autodiff_inverse_problem.png"),
            _manifest_path(output_dir / "autodiff_inverse_problem.pdf"),
        ]

    if "profiles" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "neopax_autodiff_profiles.py"),
                "--output-prefix",
                str(output_dir / "autodiff_neopax_profiles"),
            ]
        )
        manifest["profiles"] = [
            _manifest_path(output_dir / "autodiff_neopax_profiles.png"),
            _manifest_path(output_dir / "autodiff_neopax_profiles.pdf"),
        ]

    if "science" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "bootstrap_current_optimization.py"),
                "--output-prefix",
                str(output_dir / "bootstrap_current_optimization"),
            ]
        )
        manifest["science"] = [
            _manifest_path(output_dir / "bootstrap_current_optimization.png"),
            _manifest_path(output_dir / "bootstrap_current_optimization.pdf"),
        ]

    if "validation" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "validation_summary.py"),
                "--output-prefix",
                str(output_dir / "validation_summary"),
            ]
        )
        manifest["validation"] = [
            _manifest_path(output_dir / "validation_summary.png"),
            _manifest_path(output_dir / "validation_summary.pdf"),
        ]

    smoke_cpu = ROOT / "docs" / "_static" / "performance_scaling_cpu_smoke.json"
    smoke_gpu = ROOT / "docs" / "_static" / "performance_scaling_gpu_smoke.json"
    heavy_cpu = ROOT / "docs" / "_static" / "performance_scaling_cpu_heavy.json"
    heavy_gpu = ROOT / "docs" / "_static" / "performance_scaling_gpu_heavy.json"

    if "performance_smoke" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "performance_scaling.py"),
                "--cpu-json",
                str(smoke_cpu),
                "--gpu-json",
                str(smoke_gpu),
                "--figure-title",
                "Smoke-grid serial vs multiprocess scaling",
                "--output-prefix",
                str(output_dir / "performance_scaling_smoke"),
            ]
        )
        manifest["performance_smoke"] = [
            _manifest_path(output_dir / "performance_scaling_smoke.png"),
            _manifest_path(output_dir / "performance_scaling_smoke.pdf"),
        ]

    if "performance_heavy" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "performance_scaling.py"),
                "--cpu-json",
                str(heavy_cpu),
                "--gpu-json",
                str(heavy_gpu),
                "--figure-title",
                "Heavier-grid serial vs multiprocess scaling",
                "--output-prefix",
                str(output_dir / "performance_scaling_heavy"),
            ]
        )
        manifest["performance_heavy"] = [
            _manifest_path(output_dir / "performance_scaling_heavy.png"),
            _manifest_path(output_dir / "performance_scaling_heavy.pdf"),
        ]

    manifest_path = output_dir / "publication_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
