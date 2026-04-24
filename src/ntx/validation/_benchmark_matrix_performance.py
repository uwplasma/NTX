from __future__ import annotations

from ._benchmark_matrix_types import BenchmarkEntry


def performance_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
        BenchmarkEntry(
            id="performance_scaling",
            lane="performance",
            maturity="software-gate",
            title="CPU/GPU throughput characterization",
            claim_scope=(
                "Serial, multiprocess, CPU, and GPU execution modes are "
                "characterized on committed smoke and heavier-grid cases."
            ),
            literature_anchors=(
                "JAX performance practice for compiled scientific workloads",
                "multi-process scan execution for independent monoenergetic cases",
            ),
            scripts=("examples/performance_scaling.py", "scripts/benchmark_scaling.py"),
            tests=(
                "tests/test_performance_scaling_example.py",
                "tests/test_benchmark_scaling_script.py",
            ),
            artifacts=(
                "docs/_static/performance_scaling_smoke.png",
                "docs/_static/performance_scaling_smoke.pdf",
                "docs/_static/performance_scaling_cpu_smoke.json",
                "docs/_static/performance_scaling_gpu_smoke.json",
                "docs/_static/performance_scaling_heavy.png",
                "docs/_static/performance_scaling_heavy.pdf",
                "docs/_static/performance_scaling_cpu_heavy.json",
                "docs/_static/performance_scaling_gpu_heavy.json",
            ),
            manuscript_figures=("performance_scaling_heavy",),
            docs=("docs/performance.md", "docs/gpu.md"),
            open_work=(
                "map production-grid crossover points more systematically",
                "repeat the same matrix on dedicated production GPU nodes",
            ),
        ),
        BenchmarkEntry(
            id="prepared_geometry_reuse_profile",
            lane="performance",
            maturity="software-gate",
            title="Prepared-geometry and compiled-solver reuse profile",
            claim_scope=(
                "Measures repeated monoenergetic solves with direct calls, "
                "prepared geometry reuse, and a compiled prepared solver on "
                "one fixed geometry. This is a performance and reproducibility "
                "gate, not a physics-validation claim."
            ),
            literature_anchors=(
                "JAX performance practice for compiled scientific workloads",
                "reuse of fixed-geometry DKE operators across monoenergetic scans",
            ),
            scripts=("examples/prepared_geometry_reuse_profile.py",),
            tests=("tests/test_prepared_geometry_reuse_profile_example.py",),
            artifacts=(
                "docs/_static/prepared_geometry_reuse_profile.png",
                "docs/_static/prepared_geometry_reuse_profile.pdf",
                "docs/_static/prepared_geometry_reuse_profile.json",
            ),
            manuscript_figures=("prepared_geometry_reuse_profile",),
            docs=("docs/performance.md", "docs/numerics.md", "docs/manuscript.md"),
            open_work=(
                "repeat the profile on larger production VMEC-family scans",
                "evaluate reusable factorization or linear-operator approaches after profiling",
            ),
        ),
    )


__all__ = ["performance_benchmark_entries"]
