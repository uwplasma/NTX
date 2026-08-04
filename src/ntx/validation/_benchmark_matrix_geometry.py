"""Geometry rows of the benchmark matrix, at zero and finite beta.

The two geometry cases are the largest entries and share their comparison
machinery, so they sit together.
"""

from __future__ import annotations

from ._benchmark_matrix_types import BenchmarkEntry

__all__ = [
    "FINITE_BETA_GEOMETRY_BREADTH_ENTRIES",
    "geometry_breadth_benchmark_entries",
]


# --- _benchmark_matrix_geometry_finite_beta: Benchmark-matrix entries for finite-beta geometry handling. ---

FINITE_BETA_GEOMETRY_BREADTH_ENTRIES: tuple[BenchmarkEntry, ...] = (
    BenchmarkEntry(
        id="owned_geometry_neopax_dataset",
        lane="geometry-breadth",
        maturity="stress-gate",
        title="Owned finite-beta JAX-native NTX+NEOPAX dataset provenance",
        claim_scope=(
            "Builds self-contained NTX+NEOPAX scan artifacts from matching "
            "finite-beta VMEC input/wout pairs. The power-series finite-beta "
            "QA case runs through the JAX-native Boozer path, optimized "
            "finite-beta QH/QI cases are retained as direct wout-harmonic "
            "stress cases until their current-profile representation is "
            "supported by the JAX geometry stack, and all outputs store "
            "compact profile flux/current proxies from the same tables. "
            "The Boozer-coordinate path passes the physical VMEC edge "
            "toroidal flux divided by 2*pi as psi_p, closing the previous "
            "unit-flux path-normalization ambiguity. This controls geometry, "
            "interpolation, and normalization provenance; it is not an "
            "independent-code parity claim."
        ),
        literature_anchors=(
            "VMEC equilibrium representation",
            "Boozer-coordinate transport-database workflows",
            "monoenergetic coefficient database normalization used by profile solvers",
            "precise-QS and QI public VMEC example families",
        ),
        scripts=("examples/owned_geometry_neopax_dataset.py",),
        tests=("tests/test_owned_geometry_neopax_dataset.py",),
        artifacts=(
            "docs/_static/owned_geometry_neopax_dataset.png",
            "docs/_static/owned_geometry_neopax_dataset.pdf",
            "docs/_static/owned_geometry_neopax_dataset.json",
        ),
        manuscript_figures=("owned_geometry_neopax_dataset",),
        docs=(
            "README.md",
            "docs/benchmark-matrix.md",
            "docs/examples.md",
            "docs/neopax.md",
            "docs/validation.md",
        ),
        open_work=(
            (
                "compare completed same-grid SFINCS-JAX outputs across "
                "radius and collisionality before promoting new parity figures"
            ),
            (
                "add stable downstream interpolation-mode audits when those "
                "modes are exposed through a public interface"
            ),
            (
                "extend JAX geometry input reconstruction to optimized "
                "finite-beta cubic-spline current profiles"
            ),
            (
                "expand to production-resolution QA, QH, QI, and W7-X families "
                "after owned independent-code generation scripts have "
                "completed runs"
            ),
        ),
    ),
    BenchmarkEntry(
        id="owned_finite_beta_sfincs_jax_inputs",
        lane="geometry-breadth",
        maturity="stress-gate",
        title="Owned finite-beta SFINCS-JAX generation contract",
        claim_scope=(
            "Generates SFINCS-JAX RHSMode=3 input decks on the same "
            "finite-beta VMEC wout, rho, collisionality, electric-field, "
            "and resolution grids used by the owned NTX+NEOPAX scan lane. "
            "Completed outputs are ingested with the reported nu_n "
            "normalization and compared against NTX on the same geometry "
            "and grid. This is the owned independent-code generation "
            "contract, a stress-radius smoke-resolution coefficient "
            "ladder, an isolated production-grid stress-radius "
            "resolution/harmonic-cutoff probe, and a completed production "
            "radial/collisionality coefficient ladder with the exact radial "
            "interpolation, pitch-angle-scattering frequency, and RHSMode=3 "
            "flow-row normalizations recorded. It also writes a bounded "
            "RHSMode=1 profile-current diagnostic on the same finite-beta "
            "profile contract; this is not a bootstrap-current parity claim "
            "until production-resolution profile-current outputs are "
            "converged and compared against the same NTX, Redl, and profile "
            "artifacts."
        ),
        literature_anchors=(
            "SFINCS monoenergetic RHSMode=3 transport-matrix convention",
            "VMEC geometryScheme=5 finite-beta wout geometry loading",
            "same-grid independent-code validation discipline",
        ),
        scripts=(
            "examples/owned_finite_beta_sfincs_jax_inputs.py",
            "examples/owned_finite_beta_sfincs_jax_resolution_audit.py",
            "examples/owned_finite_beta_sfincs_jax_production_ladder_audit.py",
            "examples/owned_finite_beta_sfincs_jax_profile_current_audit.py",
            "examples/owned_finite_beta_sfincs_jax_profile_current_resolution_audit.py",
        ),
        tests=(
            "tests/test_owned_finite_beta_sfincs_jax_inputs.py",
            "tests/test_owned_finite_beta_sfincs_jax_resolution_audit.py",
            "tests/test_owned_finite_beta_sfincs_jax_production_ladder_audit.py",
            "tests/test_owned_finite_beta_sfincs_jax_profile_current_audit.py",
            "tests/test_owned_finite_beta_sfincs_jax_profile_current_resolution_audit.py",
        ),
        artifacts=(
            "docs/_static/owned_finite_beta_sfincs_jax_inputs.png",
            "docs/_static/owned_finite_beta_sfincs_jax_inputs.pdf",
            "docs/_static/owned_finite_beta_sfincs_jax_inputs.json",
            "docs/_static/owned_finite_beta_sfincs_jax_production_probe.png",
            "docs/_static/owned_finite_beta_sfincs_jax_production_probe.pdf",
            "docs/_static/owned_finite_beta_sfincs_jax_production_probe.json",
            "docs/_static/owned_finite_beta_sfincs_jax_production_probe_minbmn.png",
            "docs/_static/owned_finite_beta_sfincs_jax_production_probe_minbmn.pdf",
            "docs/_static/owned_finite_beta_sfincs_jax_production_probe_minbmn.json",
            "docs/_static/owned_finite_beta_sfincs_jax_production_ladder.png",
            "docs/_static/owned_finite_beta_sfincs_jax_production_ladder.pdf",
            "docs/_static/owned_finite_beta_sfincs_jax_production_ladder.json",
            "docs/_static/owned_finite_beta_sfincs_jax_production_ladder_audit.png",
            "docs/_static/owned_finite_beta_sfincs_jax_production_ladder_audit.pdf",
            "docs/_static/owned_finite_beta_sfincs_jax_production_ladder_audit.json",
            "docs/_static/owned_finite_beta_sfincs_jax_resolution_audit.png",
            "docs/_static/owned_finite_beta_sfincs_jax_resolution_audit.pdf",
            "docs/_static/owned_finite_beta_sfincs_jax_resolution_audit.json",
            "docs/_static/owned_finite_beta_sfincs_jax_profile_current_audit.png",
            "docs/_static/owned_finite_beta_sfincs_jax_profile_current_audit.pdf",
            "docs/_static/owned_finite_beta_sfincs_jax_profile_current_audit.json",
            "docs/_static/owned_finite_beta_sfincs_jax_profile_current_resolution_audit.png",
            "docs/_static/owned_finite_beta_sfincs_jax_profile_current_resolution_audit.pdf",
            "docs/_static/owned_finite_beta_sfincs_jax_profile_current_resolution_audit.json",
        ),
        manuscript_figures=(
            "owned_finite_beta_sfincs_jax_inputs",
            "owned_finite_beta_sfincs_jax_resolution_audit",
            "owned_finite_beta_sfincs_jax_production_ladder",
            "owned_finite_beta_sfincs_jax_profile_current_audit",
            "owned_finite_beta_sfincs_jax_profile_current_resolution_audit",
        ),
        docs=(
            "docs/benchmark-matrix.md",
            "docs/examples.md",
            "docs/validation.md",
        ),
        open_work=(
            (
                "keep the accepted finite-beta profile-current stress "
                "diagnostics artifact-backed as SFINCS-JAX and downstream "
                "interpolation modes evolve"
            ),
            (
                "promote only after geometry, profile, normalization, and "
                "interpolation sidecars are complete"
            ),
            (
                "treat the full-collision RHSMode=1 branch as a non-shipping "
                "feasibility diagnostic until it is practical at production "
                "resolution"
            ),
        ),
    ),
    BenchmarkEntry(
        id="owned_finite_beta_bootstrap_comparison",
        lane="geometry-breadth",
        maturity="stress-gate",
        title="Owned finite-beta Redl and NTX+NEOPAX bootstrap-current stress audit",
        claim_scope=(
            "Runs Redl and NTX+NEOPAX on the same finite-beta VMEC wout, "
            "Boozer transform, analytic profile contract, radial grid, "
            "production radial/collisionality ladder, adaptive physical "
            "nu/v support, explicit D33_spitzer audit branch, Sonine-order "
            "convergence sidecar, coefficient/profile localization sidecar, "
            "profile-current observable sidecar, current-conditioning sidecar, "
            "closure quadrature sidecar, "
            "source-channel closure sidecar, "
            "profile source-response sidecar, "
            "radial interpolation sensitivity sidecar, "
            "field-radius-matched closure quadrature sidecar, "
            "field-radius-matched source-channel sidecar, "
            "and current normalization. "
            "The current reduced-closure result has the correct sign, "
            "some radii near the 1e-1 target, and an accepted high-Nxi "
            "RHSMode=1 pitch stress gap below 1.5e-1; it is reported as "
            "a closed reduced-closure stress benchmark rather than a broad "
            "full-collision SFINCS parity claim."
        ),
        literature_anchors=(
            "Redl bootstrap-current formula and geometry-factor normalization",
            "finite-beta VMEC and Boozer-coordinate profile workflows",
            "monoenergetic database coupling to multi-species profile closures",
            "same-grid independent-code validation discipline",
        ),
        scripts=(
            "examples/owned_finite_beta_bootstrap_comparison.py",
            "examples/owned_finite_beta_closure_localization.py",
            "examples/owned_finite_beta_profile_current_observable_audit.py",
            "examples/owned_finite_beta_current_conditioning_audit.py",
            "examples/owned_finite_beta_closure_quadrature_audit.py",
            "examples/owned_finite_beta_source_channel_audit.py",
            "examples/owned_finite_beta_source_response_profile_audit.py",
            "examples/owned_finite_beta_closure_target_audit.py",
            "examples/owned_finite_beta_radial_interpolation_audit.py",
        ),
        tests=(
            "tests/test_owned_finite_beta_bootstrap_comparison.py",
            "tests/test_owned_finite_beta_closure_localization.py",
            "tests/test_owned_finite_beta_profile_current_observable_audit.py",
            "tests/test_owned_finite_beta_current_conditioning_audit.py",
            "tests/test_owned_finite_beta_closure_quadrature_audit.py",
            "tests/test_owned_finite_beta_source_channel_audit.py",
            "tests/test_owned_finite_beta_source_response_profile_audit.py",
            "tests/test_owned_finite_beta_closure_target_audit.py",
            "tests/test_owned_finite_beta_radial_interpolation_audit.py",
        ),
        artifacts=(
            "docs/_static/owned_finite_beta_bootstrap_comparison.png",
            "docs/_static/owned_finite_beta_bootstrap_comparison.pdf",
            "docs/_static/owned_finite_beta_bootstrap_comparison.json",
            "docs/_static/owned_finite_beta_closure_localization.png",
            "docs/_static/owned_finite_beta_closure_localization.pdf",
            "docs/_static/owned_finite_beta_closure_localization.json",
            "docs/_static/owned_finite_beta_profile_current_observable_audit.png",
            "docs/_static/owned_finite_beta_profile_current_observable_audit.pdf",
            "docs/_static/owned_finite_beta_profile_current_observable_audit.json",
            "docs/_static/owned_finite_beta_current_conditioning_audit.png",
            "docs/_static/owned_finite_beta_current_conditioning_audit.pdf",
            "docs/_static/owned_finite_beta_current_conditioning_audit.json",
            "docs/_static/owned_finite_beta_closure_quadrature_audit.png",
            "docs/_static/owned_finite_beta_closure_quadrature_audit.pdf",
            "docs/_static/owned_finite_beta_closure_quadrature_audit.json",
            "docs/_static/owned_finite_beta_source_channel_audit.png",
            "docs/_static/owned_finite_beta_source_channel_audit.pdf",
            "docs/_static/owned_finite_beta_source_channel_audit.json",
            "docs/_static/owned_finite_beta_source_response_profile_audit.png",
            "docs/_static/owned_finite_beta_source_response_profile_audit.pdf",
            "docs/_static/owned_finite_beta_source_response_profile_audit.json",
            "docs/_static/owned_finite_beta_closure_target_audit.png",
            "docs/_static/owned_finite_beta_closure_target_audit.pdf",
            "docs/_static/owned_finite_beta_closure_target_audit.json",
            "docs/_static/owned_finite_beta_radial_interpolation_audit.png",
            "docs/_static/owned_finite_beta_radial_interpolation_audit.pdf",
            "docs/_static/owned_finite_beta_radial_interpolation_audit.json",
            "docs/_static/owned_finite_beta_field_radius_matched_bootstrap_comparison.json",
            "docs/_static/owned_finite_beta_field_radius_matched_closure_quadrature_audit.png",
            "docs/_static/owned_finite_beta_field_radius_matched_closure_quadrature_audit.pdf",
            "docs/_static/owned_finite_beta_field_radius_matched_closure_quadrature_audit.json",
            "docs/_static/owned_finite_beta_field_radius_matched_source_channel_audit.png",
            "docs/_static/owned_finite_beta_field_radius_matched_source_channel_audit.pdf",
            "docs/_static/owned_finite_beta_field_radius_matched_source_channel_audit.json",
        ),
        manuscript_figures=(
            "owned_finite_beta_bootstrap_comparison",
            "owned_finite_beta_closure_localization",
            "owned_finite_beta_profile_current_observable_audit",
            "owned_finite_beta_current_conditioning_audit",
            "owned_finite_beta_closure_quadrature_audit",
            "owned_finite_beta_source_channel_audit",
            "owned_finite_beta_source_response_profile_audit",
            "owned_finite_beta_closure_target_audit",
            "owned_finite_beta_radial_interpolation_audit",
            "owned_finite_beta_field_radius_matched_closure_quadrature_audit",
            "owned_finite_beta_field_radius_matched_source_channel_audit",
        ),
        docs=(
            "README.md",
            "docs/benchmark-matrix.md",
            "docs/examples.md",
            "docs/neopax.md",
            "docs/validation.md",
            "docs/manuscript.md",
        ),
        open_work=(
            (
                "derive or import a quadrature-converged higher-order "
                "closure before accepting any apparent finite-beta current "
                "gate pass at X < Pmax"
            ),
            (
                "close the reduced-closure profile-current stress using the same "
                "physical profile, source-channel decomposition, normalization, "
                "and interpolation contract"
            ),
            (
                "use the radial source-response map to distinguish a "
                "physics-derived closure term from an empirical profile-local "
                "multiplier"
            ),
            (
                "repeat the field-radius-matched interpolation audit with "
                "a dense production radial scan before changing any runtime "
                "interpolation policy"
            ),
            (
                "derive a quadrature-stable finite-beta reduced-closure "
                "improvement; the current field-radius-matched sweep still "
                "has zero quadrature-stable current-gate passes"
            ),
            (
                "run the production same-grid coefficient ladder to the "
                "current-conditioned precision threshold before assigning "
                "the finite-beta net-current gap to the reduced closure"
            ),
            (
                "extend the production-resolution QA ladder to QH/QI and "
                "W7-X-owned families before promoting broad finite-beta "
                "current-profile claims"
            ),
            (
                "add downstream general-vs-legacy interpolation comparison once "
                "NEOPAX exposes a stable mode selector"
            ),
        ),
    ),
)


# --- _benchmark_matrix_geometry: Benchmark-matrix entries for geometry breadth across stellarator families. ---

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
