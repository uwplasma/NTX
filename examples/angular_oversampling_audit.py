#!/usr/bin/env python3
"""Audit angular collocation oversampling on variable-coefficient VMEC cases."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ntx import MonoenergeticCase, load_vmec_surface  # noqa: E402
from ntx._checkout_paths import find_stellopt_root, find_vmec_jax_root  # noqa: E402
from ntx.validation import audit_angular_oversampling  # noqa: E402

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "angular_oversampling_audit"
PRODUCTION_RATIOS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)
SMOKE_RATIOS = (1.0, 1.25, 1.5)


@dataclass(frozen=True)
class AuditCase:
    """One public VMEC equilibrium selected for the collocation audit."""

    id: str
    label: str
    family: str
    source: str
    path: Path


def discover_cases() -> tuple[AuditCase, ...]:
    """Discover the three public geometries used by the production audit."""

    cases: list[AuditCase] = []
    vmec_jax_root = find_vmec_jax_root()
    if vmec_jax_root is not None:
        path = vmec_jax_root / "examples/data/wout_nfp2_QA_finite_beta.nc"
        if path.exists():
            cases.append(
                AuditCase(
                    id="finite_beta_qa",
                    label="Finite-beta QA",
                    family="QA",
                    source="vmec_jax examples",
                    path=path.resolve(),
                )
            )
    stellopt_root = find_stellopt_root()
    if stellopt_root is not None:
        for case_id, label, family, filename in (
            ("ncsx", "NCSX", "compact stellarator", "wout_ncsx.nc"),
            ("hsx", "HSX", "QHS", "wout_hsx.nc"),
        ):
            path = stellopt_root / "BENCHMARKS/DIAGNO_TEST" / filename
            if path.exists():
                cases.append(
                    AuditCase(
                        id=case_id,
                        label=label,
                        family=family,
                        source="STELLOPT public benchmark",
                        path=path.resolve(),
                    )
                )
    return tuple(cases)


def _explicit_cases(paths: tuple[Path, ...]) -> tuple[AuditCase, ...]:
    return tuple(
        AuditCase(
            id=path.stem,
            label=path.stem.replace("_", " "),
            family="user-supplied",
            source="explicit CLI input",
            path=path.resolve(),
        )
        for path in paths
    )


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run_audits(
    cases: tuple[AuditCase, ...],
    *,
    ratios: tuple[float, ...],
    n_xi: int,
    recommended_oversampling: float,
    repeats: int,
) -> dict[str, object]:
    """Run all selected cases and return a machine-readable artifact payload."""

    results = []
    for case in cases:
        surface = load_vmec_surface(
            case.path,
            psi_n=0.25,
            min_bmn_to_load=1.0e-4,
        )
        audit = audit_angular_oversampling(
            surface,
            MonoenergeticCase(nu_hat=1.0e-3, er_hat=0.0),
            ratios=ratios,
            n_xi=n_xi,
            recommended_oversampling=recommended_oversampling,
            repeats=repeats,
        )
        results.append(
            {
                **asdict(case),
                "path": str(case.path),
                "theta_nyquist_floor": audit.theta_nyquist_floor,
                "zeta_nyquist_floor": audit.zeta_nyquist_floor,
                "n_xi": audit.n_xi,
                "coefficient_atol": audit.coefficient_atol,
                "recommended_oversampling": audit.recommended_oversampling,
                "points": [asdict(point) for point in audit.points],
                "recommended_max_relative_error": (audit.recommended_point.max_relative_error),
            }
        )

    max_recommended_error = max(
        float(result["recommended_max_relative_error"]) for result in results
    )
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "ntx_version": _package_version("ntx"),
            "jax_version": jax.__version__,
            "python_version": platform.python_version(),
            "git_commit": _git_commit(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "numerical_policy": {
            "nyquist_floor": "2 * max_abs_retained_mode + 1 in each angle",
            "ratios": list(ratios),
            "recommended_oversampling": recommended_oversampling,
            "reference_ratio": ratios[-1],
            "acceptance": (
                "recommendation stress metric only; research promotion still "
                "requires two successive coefficient refinements"
            ),
        },
        "cases": results,
        "summary_metrics": {
            "case_count": len(results),
            "max_recommended_relative_error": max_recommended_error,
            "recommended_error_gate": 1.0e-2,
            "recommended_error_gate_pass": max_recommended_error <= 1.0e-2,
        },
        "claim_scope": (
            "Measured variable-coefficient collocation stress diagnostic; not "
            "an analytical de-aliasing theorem or independent-code parity claim."
        ),
    }


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (14.0, 4.6),
            "figure.dpi": 220,
            "font.size": 10.0,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def plot_payload(payload: dict[str, object], output_prefix: Path) -> None:
    """Write the publication-ready error, runtime, and memory panels."""

    _configure_style()
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    cases = payload["cases"]
    assert isinstance(cases, list)
    for color, case in zip(colors, cases, strict=False):
        points = case["points"]
        ratios = np.asarray([point["requested_ratio"] for point in points])
        errors = np.maximum(
            np.asarray([point["max_relative_error"] for point in points]),
            1.0e-16,
        )
        warm = np.asarray([point["warm_execution_seconds"] for point in points])
        memory = np.asarray(
            [
                np.nan
                if point["temporary_size_bytes"] is None
                else point["temporary_size_bytes"] / 2**20
                for point in points
            ]
        )
        axes[0].semilogy(
            ratios[:-1],
            errors[:-1],
            marker="o",
            color=color,
            label=case["label"],
        )
        axes[1].plot(ratios, warm, marker="o", color=color, label=case["label"])
        axes[2].plot(ratios, memory, marker="o", color=color, label=case["label"])

    recommendation = float(payload["numerical_policy"]["recommended_oversampling"])
    axes[0].axhline(1.0e-2, color="#111827", ls="--", lw=1.1, label="1% stress gate")
    for axis in axes:
        axis.axvline(recommendation, color="#6B7280", ls=":", lw=1.2)
        axis.set_xlabel("Grid / retained-mode Nyquist floor")
    axes[0].set_ylabel("Max relative error in D11, D31, D33")
    axes[1].set_ylabel("Compiled warm execution [s]")
    axes[2].set_ylabel("XLA temporary memory [MiB]")
    axes[0].set_title("Coefficient convergence")
    axes[1].set_title("Steady execution cost")
    axes[2].set_title("Compiled temporary storage")
    axes[0].legend(fontsize=8.5)
    for label, axis in zip(("a", "b", "c"), axes, strict=True):
        axis.text(
            -0.14,
            1.02,
            f"({label})",
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
        )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"))
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="*", default=())
    parser.add_argument("--preset", choices=("smoke", "production"), default="production")
    parser.add_argument("--n-xi", type=int)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    cases = _explicit_cases(tuple(args.input)) if args.input else discover_cases()
    if not cases:
        raise FileNotFoundError("no audit inputs found; pass one or more VMEC files with --input")
    smoke = args.preset == "smoke"
    ratios = SMOKE_RATIOS if smoke else PRODUCTION_RATIOS
    recommended = 1.25 if smoke else 2.25
    n_xi = args.n_xi if args.n_xi is not None else (4 if smoke else 16)
    payload = run_audits(
        cases,
        ratios=ratios,
        n_xi=n_xi,
        recommended_oversampling=recommended,
        repeats=args.repeats,
    )
    output_prefix = args.output_prefix.resolve()
    plot_payload(payload, output_prefix)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
