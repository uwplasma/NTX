from __future__ import annotations

from ._benchmark_matrix_types import BenchmarkEntry

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
                "expand to paper-resolution QA, QH, QI, and W7-X families "
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
            "contract, an inner-radius smoke-resolution coefficient "
            "ladder, an isolated production-grid stress-radius "
            "resolution/harmonic-cutoff probe, and a completed production "
            "radial/collisionality coefficient ladder with the exact radial "
            "interpolation, pitch-angle-scattering frequency, and RHSMode=3 "
            "flow-row normalizations recorded; it is not a bootstrap-current "
            "parity claim until production-resolution profile-current outputs "
            "are compared against the same NTX, Redl, and profile artifacts."
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
        ),
        tests=(
            "tests/test_owned_finite_beta_sfincs_jax_inputs.py",
            "tests/test_owned_finite_beta_sfincs_jax_resolution_audit.py",
            "tests/test_owned_finite_beta_sfincs_jax_production_ladder_audit.py",
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
        ),
        manuscript_figures=(
            "owned_finite_beta_sfincs_jax_inputs",
            "owned_finite_beta_sfincs_jax_resolution_audit",
            "owned_finite_beta_sfincs_jax_production_ladder",
        ),
        docs=(
            "docs/benchmark-matrix.md",
            "docs/examples.md",
            "docs/validation.md",
        ),
        open_work=(
            (
                "run production SFINCS-JAX profile-current closure "
                "diagnostics against the owned finite-beta Redl and "
                "NTX+NEOPAX bootstrap-current stress audit"
            ),
            (
                "promote only after geometry, profile, normalization, and "
                "interpolation sidecars are complete"
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
            "The current reduced-closure result has the correct sign and "
            "outer-radius errors near the 1e-1 target, but explicitly records "
            "the remaining inner-radius gap; "
            "it is not promoted as SFINCS parity until same-grid SFINCS-JAX "
            "profile-current closure diagnostics are complete."
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
                "close the inner-radius reduced-closure gap using the same "
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
                "improvement; the current field-radius-matched apparent "
                "pass remains under-integrated"
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

__all__ = ["FINITE_BETA_GEOMETRY_BREADTH_ENTRIES"]
