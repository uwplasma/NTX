#!/usr/bin/env python3
"""Regenerate the manuscript-ready NTX figure bundle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIGURE_PRESETS = {
    "all": {
        "inverse",
        "profiles",
        "profile_uncertainty",
        "ambipolar",
        "ambipolar_family",
        "profile_reconstruction",
        "profile_control",
        "profile_basis",
        "profile_transport",
        "primitive_transport",
        "derivative_benchmark",
        "geometry_derivative",
        "file_backed_geometry_derivative",
        "boundary_forward_mode",
        "implicit_equilibrium_forward_mode",
        "boundary_explicit_relaxed",
        "geometry_family_breadth",
        "geometry_family_transport",
        "owned_geometry_neopax",
        "owned_finite_beta_sfincs_jax_inputs",
        "owned_finite_beta_sfincs_jax_resolution_audit",
        "owned_finite_beta_sfincs_jax_production_ladder",
        "owned_finite_beta_bootstrap_comparison",
        "owned_finite_beta_closure_localization",
        "owned_finite_beta_profile_current_observable",
        "owned_finite_beta_current_conditioning",
        "owned_finite_beta_closure_quadrature",
        "owned_finite_beta_source_channel",
        "owned_finite_beta_source_response_profile",
        "owned_finite_beta_closure_target",
        "owned_finite_beta_radial_interpolation",
        "science",
        "robust_science",
        "validation",
        "closure_validation",
        "bootstrap_proxy",
        "w7x_audit",
        "performance_smoke",
        "performance_heavy",
        "performance_production",
        "performance_strong",
        "prepared_geometry_reuse",
    },
    "main_text": {
        "validation",
        "closure_validation",
        "w7x_audit",
        "derivative_benchmark",
        "science",
        "performance_production",
        "primitive_transport",
    },
    "supplement": {
        "inverse",
        "profiles",
        "profile_uncertainty",
        "ambipolar",
        "ambipolar_family",
        "profile_reconstruction",
        "geometry_derivative",
        "file_backed_geometry_derivative",
        "boundary_forward_mode",
        "implicit_equilibrium_forward_mode",
        "boundary_explicit_relaxed",
        "geometry_family_breadth",
        "geometry_family_transport",
        "owned_geometry_neopax",
        "owned_finite_beta_sfincs_jax_inputs",
        "owned_finite_beta_sfincs_jax_resolution_audit",
        "owned_finite_beta_sfincs_jax_production_ladder",
        "owned_finite_beta_bootstrap_comparison",
        "owned_finite_beta_closure_localization",
        "owned_finite_beta_profile_current_observable",
        "owned_finite_beta_current_conditioning",
        "owned_finite_beta_closure_quadrature",
        "owned_finite_beta_source_channel",
        "owned_finite_beta_source_response_profile",
        "owned_finite_beta_closure_target",
        "owned_finite_beta_radial_interpolation",
        "profile_control",
        "profile_basis",
        "profile_transport",
        "bootstrap_proxy",
        "robust_science",
        "performance_smoke",
        "performance_heavy",
        "performance_strong",
        "prepared_geometry_reuse",
    },
}


def _run(command: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("JAX_ENABLE_X64", "1")
    subprocess.run(command, check=True, cwd=ROOT, env=env)


def _manifest_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def _copy_existing_static(prefix: str, output_dir: Path, suffixes: tuple[str, ...]) -> list[str]:
    outputs: list[str] = []
    for suffix in suffixes:
        source = ROOT / "docs" / "_static" / f"{prefix}{suffix}"
        target = output_dir / source.name
        target.write_bytes(source.read_bytes())
        outputs.append(_manifest_path(target))
    return outputs


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
        default="all",
        help="Comma-separated subset of figures to generate.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_tokens = {item.strip() for item in args.figures.split(",") if item.strip()}
    selected: set[str] = set()
    for token in selected_tokens:
        selected.update(FIGURE_PRESETS.get(token, {token}))
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

    if "profile_reconstruction" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "profile_force_reconstruction_audit.py"),
            ]
        )
        manifest["profile_reconstruction"] = _copy_existing_static(
            "profile_force_reconstruction_audit",
            output_dir,
            (".png", ".pdf", ".json"),
        )

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

    if "profile_basis" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "profile_basis_optimization.py"),
            ]
        )
        for suffix in (".png", ".pdf", ".json"):
            source = ROOT / "docs" / "_static" / f"profile_basis_optimization{suffix}"
            target = output_dir / source.name
            target.write_bytes(source.read_bytes())
        manifest["profile_basis"] = [
            _manifest_path(output_dir / "profile_basis_optimization.png"),
            _manifest_path(output_dir / "profile_basis_optimization.pdf"),
            _manifest_path(output_dir / "profile_basis_optimization.json"),
        ]

    if "profile_transport" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "profile_transport_loop.py"),
            ]
        )
        for suffix in (".png", ".pdf"):
            source = ROOT / "docs" / "_static" / f"profile_transport_loop{suffix}"
            target = output_dir / source.name
            target.write_bytes(source.read_bytes())
        manifest["profile_transport"] = [
            _manifest_path(output_dir / "profile_transport_loop.png"),
            _manifest_path(output_dir / "profile_transport_loop.pdf"),
        ]

    if "primitive_transport" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "primitive_profile_transport.py"),
            ]
        )
        for suffix in (".png", ".pdf"):
            source = ROOT / "docs" / "_static" / f"primitive_profile_transport{suffix}"
            target = output_dir / source.name
            target.write_bytes(source.read_bytes())
        manifest["primitive_transport"] = [
            _manifest_path(output_dir / "primitive_profile_transport.png"),
            _manifest_path(output_dir / "primitive_profile_transport.pdf"),
        ]

    if "derivative_benchmark" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "derivative_path_benchmark.py"),
            ]
        )
        for suffix in (".png", ".pdf", ".json"):
            source = ROOT / "docs" / "_static" / f"derivative_path_benchmark{suffix}"
            target = output_dir / source.name
            target.write_bytes(source.read_bytes())
        manifest["derivative_benchmark"] = [
            _manifest_path(output_dir / "derivative_path_benchmark.png"),
            _manifest_path(output_dir / "derivative_path_benchmark.pdf"),
            _manifest_path(output_dir / "derivative_path_benchmark.json"),
        ]

    if "geometry_derivative" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "geometry_control_derivative_benchmark.py"),
                "--output-prefix",
                str(output_dir / "geometry_control_derivative_benchmark"),
            ]
        )
        manifest["geometry_derivative"] = [
            _manifest_path(output_dir / "geometry_control_derivative_benchmark.png"),
            _manifest_path(output_dir / "geometry_control_derivative_benchmark.pdf"),
            _manifest_path(output_dir / "geometry_control_derivative_benchmark.json"),
        ]

    if "file_backed_geometry_derivative" in selected:
        manifest["file_backed_geometry_derivative"] = _copy_existing_static(
            "file_backed_geometry_control_derivative_benchmark",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "boundary_forward_mode" in selected:
        manifest["boundary_forward_mode"] = _copy_existing_static(
            "boundary_forward_mode_current_derivative_benchmark",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "implicit_equilibrium_forward_mode" in selected:
        manifest["implicit_equilibrium_forward_mode"] = _copy_existing_static(
            "implicit_equilibrium_forward_mode_derivative_benchmark",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "boundary_explicit_relaxed" in selected:
        manifest["boundary_explicit_relaxed"] = _copy_existing_static(
            "explicit_relaxed_boundary_current_derivative_benchmark",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "geometry_family_breadth" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "geometry_family_breadth_summary.py"),
                "--output-prefix",
                str(output_dir / "geometry_family_breadth_summary"),
            ]
        )
        manifest["geometry_family_breadth"] = [
            _manifest_path(output_dir / "geometry_family_breadth_summary.png"),
            _manifest_path(output_dir / "geometry_family_breadth_summary.pdf"),
            _manifest_path(output_dir / "geometry_family_breadth_summary.json"),
        ]

    if "geometry_family_transport" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "geometry_family_transport_convergence.py"),
                "--preset",
                "paper",
                "--output-prefix",
                str(output_dir / "geometry_family_transport_convergence"),
            ]
        )
        manifest["geometry_family_transport"] = [
            _manifest_path(output_dir / "geometry_family_transport_convergence.png"),
            _manifest_path(output_dir / "geometry_family_transport_convergence.pdf"),
            _manifest_path(output_dir / "geometry_family_transport_convergence.json"),
        ]

    if "owned_geometry_neopax" in selected:
        manifest["owned_geometry_neopax"] = _copy_existing_static(
            "owned_geometry_neopax_dataset",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "owned_finite_beta_sfincs_jax_inputs" in selected:
        manifest["owned_finite_beta_sfincs_jax_inputs"] = _copy_existing_static(
            "owned_finite_beta_sfincs_jax_inputs",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "owned_finite_beta_sfincs_jax_resolution_audit" in selected:
        manifest["owned_finite_beta_sfincs_jax_resolution_audit"] = _copy_existing_static(
            "owned_finite_beta_sfincs_jax_resolution_audit",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "owned_finite_beta_sfincs_jax_production_ladder" in selected:
        manifest["owned_finite_beta_sfincs_jax_production_ladder"] = _copy_existing_static(
            "owned_finite_beta_sfincs_jax_production_ladder_audit",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "owned_finite_beta_bootstrap_comparison" in selected:
        manifest["owned_finite_beta_bootstrap_comparison"] = _copy_existing_static(
            "owned_finite_beta_bootstrap_comparison",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "owned_finite_beta_closure_localization" in selected:
        manifest["owned_finite_beta_closure_localization"] = _copy_existing_static(
            "owned_finite_beta_closure_localization",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "owned_finite_beta_profile_current_observable" in selected:
        manifest["owned_finite_beta_profile_current_observable"] = _copy_existing_static(
            "owned_finite_beta_profile_current_observable_audit",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "owned_finite_beta_current_conditioning" in selected:
        manifest["owned_finite_beta_current_conditioning"] = _copy_existing_static(
            "owned_finite_beta_current_conditioning_audit",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "owned_finite_beta_closure_quadrature" in selected:
        manifest["owned_finite_beta_closure_quadrature"] = _copy_existing_static(
            "owned_finite_beta_closure_quadrature_audit",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "owned_finite_beta_source_channel" in selected:
        manifest["owned_finite_beta_source_channel"] = _copy_existing_static(
            "owned_finite_beta_source_channel_audit",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "owned_finite_beta_source_response_profile" in selected:
        manifest["owned_finite_beta_source_response_profile"] = _copy_existing_static(
            "owned_finite_beta_source_response_profile_audit",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "owned_finite_beta_closure_target" in selected:
        manifest["owned_finite_beta_closure_target"] = _copy_existing_static(
            "owned_finite_beta_closure_target_audit",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "owned_finite_beta_radial_interpolation" in selected:
        manifest["owned_finite_beta_radial_interpolation"] = _copy_existing_static(
            "owned_finite_beta_radial_interpolation_audit",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    if "profile_uncertainty" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "autodiff_profile_uncertainty.py"),
                "--output-prefix",
                str(output_dir / "autodiff_profile_uncertainty"),
            ]
        )
        manifest["profile_uncertainty"] = [
            _manifest_path(output_dir / "autodiff_profile_uncertainty.png"),
            _manifest_path(output_dir / "autodiff_profile_uncertainty.pdf"),
            _manifest_path(output_dir / "autodiff_profile_uncertainty.json"),
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
            _manifest_path(output_dir / "bootstrap_current_optimization.json"),
        ]

    if "robust_science" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "bootstrap_current_robust_optimization.py"),
                "--output-prefix",
                str(output_dir / "bootstrap_current_robust_optimization"),
            ]
        )
        manifest["robust_science"] = [
            _manifest_path(output_dir / "bootstrap_current_robust_optimization.png"),
            _manifest_path(output_dir / "bootstrap_current_robust_optimization.pdf"),
            _manifest_path(output_dir / "bootstrap_current_robust_optimization.json"),
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
            _manifest_path(output_dir / "validation_summary.json"),
        ]

    if "closure_validation" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_closure_validation_report.py"),
            ]
        )
        manifest["closure_validation"] = _copy_existing_static(
            "closure_validation_report",
            output_dir,
            (".png", ".pdf", ".json", ".txt"),
        )

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
        manifest["w7x_audit"] = _copy_existing_static(
            "bootstrap_current_reference_audit_w7x",
            output_dir,
            (".png", ".pdf", ".json"),
        )

    smoke_cpu = ROOT / "docs" / "_static" / "performance_scaling_cpu_smoke.json"
    smoke_gpu = ROOT / "docs" / "_static" / "performance_scaling_gpu_smoke.json"
    heavy_cpu = ROOT / "docs" / "_static" / "performance_scaling_cpu_heavy.json"
    heavy_gpu = ROOT / "docs" / "_static" / "performance_scaling_gpu_heavy.json"
    production_cpu = ROOT / "docs" / "_static" / "performance_scaling_cpu_production.json"
    production_gpu = ROOT / "docs" / "_static" / "performance_scaling_gpu_production.json"
    strong_cpu = ROOT / "docs" / "_static" / "performance_strong_scaling_cpu_production.json"
    strong_gpu = ROOT / "docs" / "_static" / "performance_strong_scaling_gpu_production.json"

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

    if "performance_production" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "performance_scaling.py"),
                "--cpu-json",
                str(production_cpu),
                "--gpu-json",
                str(production_gpu),
                "--figure-title",
                "Production-grid serial vs parallel scaling",
                "--output-prefix",
                str(output_dir / "performance_scaling_production"),
            ]
        )
        manifest["performance_production"] = [
            _manifest_path(output_dir / "performance_scaling_production.png"),
            _manifest_path(output_dir / "performance_scaling_production.pdf"),
            _manifest_path(output_dir / "performance_scaling_production.json"),
        ]

    if "performance_strong" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "performance_strong_scaling.py"),
                "--cpu-json",
                str(strong_cpu),
                "--gpu-json",
                str(strong_gpu),
                "--figure-title",
                "Production fixed-workload strong scaling",
                "--output-prefix",
                str(output_dir / "performance_strong_scaling_production"),
            ]
        )
        manifest["performance_strong"] = [
            _manifest_path(output_dir / "performance_strong_scaling_production.png"),
            _manifest_path(output_dir / "performance_strong_scaling_production.pdf"),
            _manifest_path(output_dir / "performance_strong_scaling_production.json"),
        ]

    if "prepared_geometry_reuse" in selected:
        _run(
            [
                sys.executable,
                str(ROOT / "examples" / "prepared_geometry_reuse_profile.py"),
                "--preset",
                "paper",
                "--output-prefix",
                str(output_dir / "prepared_geometry_reuse_profile"),
            ]
        )
        manifest["prepared_geometry_reuse"] = [
            _manifest_path(output_dir / "prepared_geometry_reuse_profile.png"),
            _manifest_path(output_dir / "prepared_geometry_reuse_profile.pdf"),
            _manifest_path(output_dir / "prepared_geometry_reuse_profile.json"),
        ]

    manifest_path = output_dir / "publication_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
