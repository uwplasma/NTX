from __future__ import annotations

from ._benchmark_matrix_geometry_finite_beta import FINITE_BETA_GEOMETRY_BREADTH_ENTRIES
from ._benchmark_matrix_types import BenchmarkEntry


def geometry_breadth_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
        BenchmarkEntry(
            id="geometry_family_breadth_summary",
            lane="geometry-breadth",
            maturity="stress-gate",
            title="Artifact-backed geometry-family derivative breadth summary",
            claim_scope=(
                "Summarizes committed analytic, file-backed, boundary-projected, "
                "explicit-relaxed, and implicit-equilibrium diagnostic artifacts "
                "without promoting a full hidden-symmetry, omnigenous, or broad "
                "W7-X/QI validation claim."
            ),
            literature_anchors=(
                "Paul et al. 2019 adjoint neoclassical optimization",
                "McGreivy 2024 differentiable programming for plasma workflows",
                "Landreman and Paul 2022 precise-QS benchmark family",
                "omnigenous and quasi-isodynamic geometry-breadth literature",
            ),
            scripts=("examples/geometry_family_breadth_summary.py",),
            tests=("tests/test_geometry_family_breadth_summary.py",),
            artifacts=(
                "docs/_static/geometry_family_breadth_summary.png",
                "docs/_static/geometry_family_breadth_summary.pdf",
                "docs/_static/geometry_family_breadth_summary.json",
            ),
            manuscript_figures=("geometry_family_breadth_summary",),
            docs=(
                "docs/benchmark-matrix.md",
                "docs/autodiff.md",
                "docs/manuscript.md",
                "docs/research-roadmap.md",
            ),
            open_work=(
                "broaden committed cases to reusable W7-X EIM/KJM, QI, and omnigenous inputs",
                (
                    "add D11/D31/D33 parity and convergence ladders before full "
                    "geometry-family promotion"
                ),
                (
                    "restore implicit-equilibrium derivatives only after residual "
                    "contraction and surface/transport tangent parity pass"
                ),
            ),
        ),
        BenchmarkEntry(
            id="geometry_family_transport_convergence",
            lane="geometry-breadth",
            maturity="stress-gate",
            title="VMEC geometry-family D11/D31/D33 convergence stress diagnostic",
            claim_scope=(
                "Runs reusable public VMEC examples through NTX and reports "
                "D11, D31, and D33 coarse-to-fine changes. This closes a "
                "broad reduced-grid NTX stress diagnostic, not an "
                "independent-code parity claim."
            ),
            literature_anchors=(
                "W7-X standard-configuration benchmark workflows",
                "Landreman and Paul 2022 precise-QS benchmark family",
                "quasi-isodynamic and omnigenous geometry-family validation literature",
                "VMEC, STELLOPT, and SIMSOPT public equilibrium example suites",
            ),
            scripts=("examples/geometry_family_transport_convergence.py",),
            tests=("tests/test_geometry_family_transport_convergence.py",),
            artifacts=(
                "docs/_static/geometry_family_transport_convergence.png",
                "docs/_static/geometry_family_transport_convergence.pdf",
                "docs/_static/geometry_family_transport_convergence.json",
            ),
            manuscript_figures=("geometry_family_transport_convergence",),
            docs=(
                "README.md",
                "docs/benchmark-matrix.md",
                "docs/manuscript.md",
                "docs/research-roadmap.md",
                "docs/validation.md",
            ),
            open_work=(
                (
                    "promote only after production-resolution sweeps with "
                    "independent reference parity on each family"
                ),
                (
                    "add an owned W7-X KJM input once a reusable public "
                    "reference input is identified or regenerated"
                ),
                (
                    "add radial and collisionality ladders before claiming "
                    "broad bootstrap-current-profile validation"
                ),
            ),
        ),
        BenchmarkEntry(
            id="angular_oversampling_audit",
            lane="geometry-breadth",
            maturity="stress-gate",
            title="Variable-coefficient angular collocation oversampling audit",
            claim_scope=(
                "Measures D11, D31, and D33 error, compiled warm runtime, and "
                "XLA temporary memory against a finer collocation reference. "
                "It supports a warning-level starting-grid recommendation, not "
                "an analytical de-aliasing theorem or independent-code claim."
            ),
            literature_anchors=(
                "Orszag 1971 Fourier alias-elimination analysis",
                "Escoto et al. 2024 monoenergetic Fourier-collocation convergence",
                "Escoto 2025 thesis angular-resolution practice",
            ),
            scripts=("examples/angular_oversampling_audit.py",),
            tests=("tests/test_angular_oversampling.py",),
            artifacts=(
                "docs/_static/angular_oversampling_audit.png",
                "docs/_static/angular_oversampling_audit.pdf",
                "docs/_static/angular_oversampling_audit.json",
            ),
            manuscript_figures=("angular_oversampling_audit",),
            docs=(
                "docs/convergence.md",
                "docs/examples.md",
                "docs/literature.md",
                "docs/numerics.md",
                "docs/testing.md",
                "docs/validation.md",
            ),
            open_work=(
                "retain two-successive-grid convergence as the research gate",
                "extend the measured recommendation if new geometry families exceed it",
            ),
        ),
        BenchmarkEntry(
            id="boozmn_same_coordinate_roundtrip",
            lane="geometry-breadth",
            maturity="positive-gate",
            title="Same-coordinate Boozer-file round-trip validation",
            claim_scope=(
                "Generates a Boozer file from a VMEC wout, reloads the same "
                "VMEC half-grid surfaces through the direct boozmn backend, "
                "and requires geometry metadata plus D11/D31/D13/D33 to match "
                "the in-memory vmex/booz_xform_jax path. This validates "
                "the direct loader radial coordinate and normalization "
                "conventions; it does not equate VMEC-harmonic and "
                "Boozer-coordinate representations."
            ),
            literature_anchors=(
                "Boozer-coordinate flux-surface representation",
                "VMEC half-grid placement of magnetic-field spectra",
                "Boozer-transform boozmn jlist and packed-surface convention",
            ),
            scripts=("examples/boozmn_same_coordinate_roundtrip_audit.py",),
            tests=(
                "tests/test_boozmn.py",
                "tests/test_boozmn_same_coordinate_roundtrip_audit.py",
            ),
            artifacts=(
                "docs/_static/boozmn_same_coordinate_roundtrip_audit.png",
                "docs/_static/boozmn_same_coordinate_roundtrip_audit.pdf",
                "docs/_static/boozmn_same_coordinate_roundtrip_audit.json",
            ),
            manuscript_figures=("boozmn_same_coordinate_roundtrip_audit",),
            docs=(
                "docs/geometry.md",
                "docs/neopax.md",
                "docs/validation.md",
                "docs/benchmark-matrix.md",
            ),
            open_work=(
                (
                    "keep VMEC-harmonic versus Boozer-file comparisons scoped "
                    "as representation audits unless their source channels are "
                    "shown to be mathematically identical"
                ),
                (
                    "repeat the round-trip gate on larger finite-beta family "
                    "inputs when those artifacts are promoted"
                ),
            ),
        ),
        BenchmarkEntry(
            id="boozmn_finite_beta_wout_roundtrip",
            lane="geometry-breadth",
            maturity="positive-gate",
            title="Finite-beta finalized-wout Boozer-file transfer validation",
            claim_scope=(
                "Transforms an optimized finite-beta QA VMEC wout through "
                "finalized magnetic channels, reloads the generated Boozer "
                "file on the same half-grid surfaces, and requires "
                "D11/D31/D13/D33 to match the reference transform to "
                "roundoff. This validates the file-backed finite-beta "
                "Boozer path for input profile representations that the "
                "differentiable VMEC-state reconstruction path does not yet "
                "support; it is not a claim of differentiable finite-beta "
                "state sensitivities."
            ),
            literature_anchors=(
                "VMEC finite-beta wout magnetic-field representation",
                "Boozer-coordinate flux-surface representation",
                "VMEC half-grid placement of Boozer spectra and radial profiles",
                "finite-beta quasi-symmetric benchmark input families",
            ),
            scripts=("examples/boozmn_same_coordinate_roundtrip_audit.py",),
            tests=(
                "tests/test_boozmn.py",
                "tests/test_boozmn_same_coordinate_roundtrip_audit.py",
                "tests/test_vmex_backend.py",
            ),
            artifacts=(
                "docs/_static/boozmn_finite_beta_wout_roundtrip_audit.png",
                "docs/_static/boozmn_finite_beta_wout_roundtrip_audit.pdf",
                "docs/_static/boozmn_finite_beta_wout_roundtrip_audit.json",
            ),
            manuscript_figures=("boozmn_finite_beta_wout_roundtrip_audit",),
            docs=(
                "docs/geometry.md",
                "docs/validation.md",
                "docs/testing.md",
                "docs/physics-gates.md",
                "docs/benchmark-matrix.md",
            ),
            open_work=(
                (
                    "promote fully differentiable finite-beta state sensitivities "
                    "only after the optional VMEC stack supports the optimized "
                    "current-profile representation or exposes solver-consistent "
                    "finalized magnetic channels in differentiable form"
                ),
                (
                    "extend the same finalized-wout transfer gate to QH, QI, "
                    "and W7-X-owned finite-beta cases when those owned inputs "
                    "are promoted"
                ),
            ),
        ),
        *FINITE_BETA_GEOMETRY_BREADTH_ENTRIES,
        BenchmarkEntry(
            id="geometry_breadth_hidden_symmetry",
            lane="geometry-breadth",
            maturity="planned-lane",
            title="Hidden-symmetry and omnigenous geometry families",
            claim_scope=(
                "Future research workflows should broaden validation beyond the "
                "current W7-X-centered set."
            ),
            literature_anchors=(
                "near-axis quasi-isodynamic construction and verification",
                "hidden-symmetry optimization literature",
                "piecewise-omnigenous optimization literature",
            ),
            scripts=(),
            tests=(),
            artifacts=(),
            manuscript_figures=(),
            docs=("docs/research-roadmap.md",),
            open_work=(
                "identify reusable public inputs for hidden-symmetry studies",
                "add VMEC/Boozer family examples once inputs are owned",
                "define convergence gates before promoting figures",
            ),
        ),
    )


__all__ = ["geometry_breadth_benchmark_entries"]
