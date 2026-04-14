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
        default=(
            "inverse,profiles,ambipolar,ambipolar_family,profile_control,derivative_benchmark,science,validation,"
            "bootstrap_proxy,w7x_audit,performance_smoke,performance_heavy"
        ),
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

    if "ambipolar" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "ambipolar_profile.py"),
            ]
        )
        for suffix in (".png", ".pdf"):
            source = ROOT / "docs" / "_static" / f"ambipolar_profile{suffix}"
            target = output_dir / source.name
            target.write_bytes(source.read_bytes())
        manifest["ambipolar"] = [
            _manifest_path(output_dir / "ambipolar_profile.png"),
            _manifest_path(output_dir / "ambipolar_profile.pdf"),
        ]

    if "ambipolar_family" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "ambipolar_profile_family.py"),
            ]
        )
        for suffix in (".png", ".pdf"):
            source = ROOT / "docs" / "_static" / f"ambipolar_profile_family{suffix}"
            target = output_dir / source.name
            target.write_bytes(source.read_bytes())
        manifest["ambipolar_family"] = [
            _manifest_path(output_dir / "ambipolar_profile_family.png"),
            _manifest_path(output_dir / "ambipolar_profile_family.pdf"),
        ]

    if "profile_control" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "profile_control_optimization.py"),
            ]
        )
        for suffix in (".png", ".pdf"):
            source = ROOT / "docs" / "_static" / f"profile_control_optimization{suffix}"
            target = output_dir / source.name
            target.write_bytes(source.read_bytes())
        manifest["profile_control"] = [
            _manifest_path(output_dir / "profile_control_optimization.png"),
            _manifest_path(output_dir / "profile_control_optimization.pdf"),
        ]

    if "derivative_benchmark" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "derivative_path_benchmark.py"),
            ]
        )
        for suffix in (".png", ".pdf"):
            source = ROOT / "docs" / "_static" / f"derivative_path_benchmark{suffix}"
            target = output_dir / source.name
            target.write_bytes(source.read_bytes())
        manifest["derivative_benchmark"] = [
            _manifest_path(output_dir / "derivative_path_benchmark.png"),
            _manifest_path(output_dir / "derivative_path_benchmark.pdf"),
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

    if "bootstrap_proxy" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "bootstrap_current_from_vmec_or_boozmn.py"),
            ]
        )
        for suffix in (".png", ".pdf", ".json"):
            source = ROOT / "docs" / "_static" / f"bootstrap_current_from_vmec_or_boozmn{suffix}"
            target = output_dir / source.name
            target.write_bytes(source.read_bytes())
        manifest["bootstrap_proxy"] = [
            _manifest_path(output_dir / "bootstrap_current_from_vmec_or_boozmn.png"),
            _manifest_path(output_dir / "bootstrap_current_from_vmec_or_boozmn.pdf"),
            _manifest_path(output_dir / "bootstrap_current_from_vmec_or_boozmn.json"),
        ]

    if "w7x_audit" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "bootstrap_current_reference_audit_w7x.py"),
            ]
        )
        for suffix in (".png", ".pdf", ".json"):
            source = ROOT / "docs" / "_static" / f"bootstrap_current_reference_audit_w7x{suffix}"
            target = output_dir / source.name
            target.write_bytes(source.read_bytes())
        manifest["w7x_audit"] = [
            _manifest_path(output_dir / "bootstrap_current_reference_audit_w7x.png"),
            _manifest_path(output_dir / "bootstrap_current_reference_audit_w7x.pdf"),
            _manifest_path(output_dir / "bootstrap_current_reference_audit_w7x.json"),
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
