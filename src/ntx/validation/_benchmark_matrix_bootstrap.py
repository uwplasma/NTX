from __future__ import annotations

from ._benchmark_matrix_types import BenchmarkEntry


def bootstrap_current_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
        BenchmarkEntry(
            id="precise_qs_redl_sfincs",
            lane="bootstrap-current",
            maturity="positive-gate",
            title="Precise-QS Redl current against archived fixed-field reference",
            claim_scope=(
                "The Redl path closes the archived precise-QS fixed-field "
                "interior-window gate and is separated from the reduced closure "
                "stress metric."
            ),
            literature_anchors=(
                "Landreman and Paul 2022 precise-QS benchmark family",
                "Redl et al. bootstrap-current fit",
                "SFINCS fixed-field archive used by the benchmark",
            ),
            scripts=(
                "examples/precise_qs_redl_sfincs_audit.py",
                "examples/bootstrap_current_fixed_field_validation.py",
                "scripts/build_closure_validation_report.py",
            ),
            tests=(
                "tests/test_precise_qs_redl_sfincs_audit.py",
                "tests/test_fixed_field_parallel_flow_audit.py",
            ),
            artifacts=(
                "docs/_static/bootstrap_current_fixed_field_validation.png",
                "docs/_static/bootstrap_current_fixed_field_validation.pdf",
                "docs/_static/bootstrap_current_fixed_field_validation.json",
                "docs/_static/closure_validation_report.png",
                "docs/_static/closure_validation_report.pdf",
                "docs/_static/closure_validation_report.json",
                "docs/_static/closure_validation_report.txt",
            ),
            manuscript_figures=("closure_validation_report",),
            docs=("docs/physics-gates.md", "docs/validation.md"),
        ),
        BenchmarkEntry(
            id="fixed_field_ntx_neopax_closure_stress",
            lane="bootstrap-current",
            maturity="stress-gate",
            title="Fixed-field current closure stress test",
            claim_scope=(
                "The fixed-field NTX+NEOPAX total-current comparison passes the "
                "documented reduced-closure stress gate with the low-order "
                "Spitzer-conductivity branch; species-current parity and broader "
                "closure defaults remain separate claims."
            ),
            literature_anchors=(
                "Landreman and Paul 2022 precise-QS benchmark family",
                "momentum-restoring closure literature for parallel-flow models",
            ),
            scripts=(
                "examples/bootstrap_current_fixed_field_validation.py",
                "examples/fixed_field_momentum_correction_diagnostic.py",
                "examples/momentum_correction_mapping_audit.py",
                "scripts/build_closure_validation_report.py",
            ),
            tests=(
                "tests/test_fixed_field_momentum_correction_diagnostic.py",
                "tests/test_momentum_correction_mapping_audit.py",
            ),
            artifacts=(
                "docs/_static/bootstrap_current_fixed_field_validation.png",
                "docs/_static/bootstrap_current_fixed_field_validation.pdf",
                "docs/_static/bootstrap_current_fixed_field_validation.json",
                "docs/_static/closure_validation_report.png",
                "docs/_static/closure_validation_report.pdf",
                "docs/_static/closure_validation_report.json",
                "docs/_static/closure_validation_report.txt",
                "docs/_static/closure_pmax_convergence.png",
                "docs/_static/closure_pmax_convergence.pdf",
                "docs/_static/closure_pmax_convergence.json",
            ),
            manuscript_figures=("closure_validation_report",),
            docs=("docs/physics-gates.md", "docs/validation.md"),
            open_work=(
                "derive any broader default from the same moment-equation closure",
                "keep the integrated W7-X raw-branch transfer regression separate",
            ),
        ),
    )


__all__ = ["bootstrap_current_benchmark_entries"]
