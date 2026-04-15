#!/usr/bin/env python3
"""Build manuscript tables and reproducibility metadata from NTX artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "docs" / "_static"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def _format_float(value: float, scientific: bool = False) -> str:
    if scientific:
        return f"{value:.3e}"
    return f"{value:.3f}"


def build_payload() -> dict:
    w7x = _load_json(STATIC / "bootstrap_current_reference_audit_w7x.json")
    derivative = _load_json(STATIC / "derivative_path_benchmark.json")
    science = _load_json(STATIC / "bootstrap_current_optimization.json")
    cpu = _load_json(STATIC / "performance_scaling_cpu_heavy.json")
    gpu = _load_json(STATIC / "performance_scaling_gpu_heavy.json")
    figures = _load_json(STATIC / "publication_figure_manifest.json")
    main_text = [
        "validation",
        "w7x_audit",
        "derivative_benchmark",
        "science",
        "performance_heavy",
        "primitive_transport",
    ]
    supplement = [
        "inverse",
        "profiles",
        "ambipolar",
        "ambipolar_family",
        "profile_control",
        "profile_basis",
        "profile_transport",
        "bootstrap_proxy",
        "performance_smoke",
    ]
    fine_w7x_error = w7x["bootstrap_current_errors"][-1]["max_relative_error"]
    cpu_best = max(
        row["multiprocess_speedup_vs_serial"] for row in cpu["results"]
    )
    gpu_best = max(
        row["multiprocess_speedup_vs_serial"] for row in gpu["results"]
    )

    return {
        "git": {
            "commit": _git_output("rev-parse", "HEAD"),
            "branch": _git_output("branch", "--show-current"),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
        },
        "tables": {
            "validation": {
                "bootstrap_current_reference_scale": w7x["bootstrap_current_reference_scale"],
                "bootstrap_current_errors": w7x["bootstrap_current_errors"],
            },
            "derivatives": {
                "grid": derivative["grid"],
                "nu_hat": derivative["nu_hat"],
                "er_min": derivative["er_min"],
                "er_max": derivative["er_max"],
                "max_relative_mismatch": max(derivative["max_relative_mismatch"]),
                "best_prepared_speedup": max(derivative["speedup_prepared_vs_direct"]),
            },
            "performance": {
                "cpu_heavy": cpu,
                "gpu_heavy": gpu,
            },
            "science": {
                "wout": science["wout"],
                "harmonic_m": science["harmonic_m"],
                "harmonic_n": science["harmonic_n"],
                "baseline_scale": science["baseline_scale"],
                "optimized_scale": science["optimized_scale"],
                "weighted_gain": science["weighted_gain"],
                "serial_scan_seconds": science["serial_scan_seconds"],
                "parallel_scan_seconds": science["parallel_scan_seconds"],
            },
        },
        "claims": {
            "w7x_fine_grid_max_relative_error": fine_w7x_error,
            "derivative_max_relative_mismatch": max(derivative["max_relative_mismatch"]),
            "best_prepared_derivative_speedup": max(derivative["speedup_prepared_vs_direct"]),
            "bootstrap_current_weighted_gain": science["weighted_gain"],
            "cpu_heavy_best_multiprocess_speedup": cpu_best,
            "gpu_heavy_best_multiprocess_speedup": gpu_best,
            "gpu_heavy_healthy_device_count": gpu["healthy_parallel_device_count"],
        },
        "figures": figures,
        "figure_sets": {
            "main_text": main_text,
            "supplement": supplement,
        },
        "commands": {
            "figure_bundle": "python examples/make_publication_figures.py --figures main_text,supplement",
            "main_text_figures": "python examples/make_publication_figures.py --figures main_text",
            "supplement_figures": "python examples/make_publication_figures.py --figures supplement",
            "tables": "python scripts/build_manuscript_artifacts.py",
            "validation_subset": (
                "python -m pytest -q "
                "tests/test_w7x_reference_benchmark.py "
                "tests/test_derivative_path_benchmark_example.py "
                "tests/test_bootstrap_current_optimization_example.py "
                "tests/test_manuscript_artifacts_script.py "
                "tests/test_make_publication_figures.py -k "
                "\"subset_writes_manifest or bootstrap_subset_writes_manifest\""
            ),
        },
    }


def build_markdown(payload: dict) -> str:
    validation_rows = payload["tables"]["validation"]["bootstrap_current_errors"]
    cpu_rows = payload["tables"]["performance"]["cpu_heavy"]["results"]
    gpu_rows = payload["tables"]["performance"]["gpu_heavy"]["results"]
    science = payload["tables"]["science"]
    derivatives = payload["tables"]["derivatives"]

    lines = [
        "# NTX Manuscript Tables",
        "",
        "## Validation",
        "",
        "| Grid `(N_theta, N_zeta, N_xi)` | Max relative error |",
        "| --- | ---: |",
    ]
    for row in validation_rows:
        grid = tuple(row["grid"])
        lines.append(f"| `{grid}` | {_format_float(row['max_relative_error'], scientific=True)} |")

    lines.extend(
        [
            "",
            "## Derivatives",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            f"| Grid | `{tuple(derivatives['grid'].values())}` |",
            f"| `nu_hat` | `{derivatives['nu_hat']:.3e}` |",
            f"| `E_r` scan | `{derivatives['er_min']:.3e}` to `{derivatives['er_max']:.3e}` |",
            f"| Max relative mismatch | `{derivatives['max_relative_mismatch']:.3e}` |",
            f"| Best prepared speedup | `{derivatives['best_prepared_speedup']:.3f}x` |",
            "",
            "## Bootstrap-Current Optimization",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            f"| Harmonic `(m, n)` | `({science['harmonic_m']}, {science['harmonic_n']})` |",
            f"| Baseline scale | `{science['baseline_scale']:.3f}` |",
            f"| Optimized scale | `{science['optimized_scale']:.3f}` |",
            f"| Weighted current gain | `{science['weighted_gain']:.3f}x` |",
            f"| Serial scan time | `{science['serial_scan_seconds']:.3f} s` |",
            f"| Parallel scan time | `{science['parallel_scan_seconds']:.3f} s` |",
            "",
            "## Performance",
            "",
            "### CPU heavy-grid scaling",
            "",
            "| Cases | Serial [s] | Multiprocess [s] | Speedup |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in cpu_rows:
        lines.append(
            f"| {row['num_cases']} | {_format_float(row['serial_seconds'])} | "
            f"{_format_float(row['multiprocess_seconds'])} | {_format_float(row['multiprocess_speedup_vs_serial'])}x |"
        )

    lines.extend(
        [
            "",
            "### GPU heavy-grid scaling",
            "",
            "| Cases | Serial [s] | Multiprocess [s] | Speedup | Healthy devices |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in gpu_rows:
        lines.append(
            f"| {row['num_cases']} | {_format_float(row['serial_seconds'])} | "
            f"{_format_float(row['multiprocess_seconds'])} | {_format_float(row['multiprocess_speedup_vs_serial'])}x | "
            f"{payload['tables']['performance']['gpu_heavy']['healthy_parallel_device_count']} |"
        )

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "| Key | Value |",
            "| --- | --- |",
            f"| Commit | `{payload['git']['commit']}` |",
            f"| Branch | `{payload['git']['branch']}` |",
            f"| Python | `{payload['environment']['python']}` |",
            f"| JAX | `{payload['environment']['jax']}` |",
            f"| NumPy | `{payload['environment']['numpy']}` |",
            f"| Platform | `{payload['environment']['platform']}` |",
            f"| Figure bundle | `{payload['commands']['figure_bundle']}` |",
            f"| Main-text figures | `{payload['commands']['main_text_figures']}` |",
            f"| Supplement figures | `{payload['commands']['supplement_figures']}` |",
            f"| Artifact tables | `{payload['commands']['tables']}` |",
            f"| Validation subset | `{payload['commands']['validation_subset']}` |",
            "",
        ]
    )
    return "\n".join(lines)


def build_claims_markdown(payload: dict) -> str:
    claims = payload["claims"]
    return "\n".join(
        [
            "# NTX Manuscript Claims",
            "",
            "These are the current paper-facing technical claims derived directly from the",
            "validated NTX artifacts.",
            "",
            f"- W7-X imported-workflow bootstrap-current convergence reaches a maximum relative error of `{claims['w7x_fine_grid_max_relative_error']:.3e}` on the fine `25 x 25 x 64` grid.",
            f"- The prepared implicit-adjoint derivative path matches direct reverse-mode with a maximum relative mismatch of `{claims['derivative_max_relative_mismatch']:.3e}` on the committed derivative benchmark.",
            f"- The prepared derivative path reaches a best observed speedup of `{claims['best_prepared_derivative_speedup']:.3f}x` on the benchmarked electric-field scan.",
            f"- The differentiable bootstrap-current optimization example improves the weighted current proxy by `{claims['bootstrap_current_weighted_gain']:.3f}x` on the committed W7-X study.",
            f"- On the heavy CPU benchmark, multiprocess execution reaches a best observed speedup of `{claims['cpu_heavy_best_multiprocess_speedup']:.3f}x`.",
            f"- On the heavy GPU benchmark, the current multiprocess path reaches a best observed speedup of `{claims['gpu_heavy_best_multiprocess_speedup']:.3f}x` with `{claims['gpu_heavy_healthy_device_count']}` healthy parallel GPU device(s), so the current paper should frame GPU multiprocess as a characterized execution mode rather than a throughput win.",
            "",
            "These claims should be used consistently in the manuscript text, captions, and",
            "response-to-reviewer notes.",
            "",
        ]
    )


def main() -> None:
    payload = build_payload()
    markdown = build_markdown(payload)
    claims = build_claims_markdown(payload)
    (STATIC / "manuscript_artifacts.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (STATIC / "manuscript_tables.md").write_text(markdown, encoding="utf-8")
    (STATIC / "manuscript_claims.md").write_text(claims, encoding="utf-8")
    print(f"Wrote {STATIC / 'manuscript_artifacts.json'}")
    print(f"Wrote {STATIC / 'manuscript_tables.md'}")
    print(f"Wrote {STATIC / 'manuscript_claims.md'}")


if __name__ == "__main__":
    main()
