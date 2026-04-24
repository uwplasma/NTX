from __future__ import annotations

from ._physics_gate_types import PhysicsGate

ANALYTICAL_GATES: tuple[PhysicsGate, ...] = (
    PhysicsGate(
        name="onsager_symmetry",
        category="analytical",
        metric="|D13 + D31|",
        relation="test",
        threshold=None,
        source="tests/test_solver.py and examples/validation_summary.py",
        rationale=(
            "The monoenergetic solve must preserve the Onsager symmetry expected "
            "for the source split and the Legendre-space discretization."
        ),
    ),
    PhysicsGate(
        name="p2_projection_exact_recovery",
        category="analytical",
        metric="generated Sonine/Hankel P=2 recovery",
        relation="<=",
        threshold=1.0e-12,
        source="local imported closure tests/test_moment_projection.py",
        rationale=(
            "Any higher-order closure work must reduce exactly to the present "
            "three-moment system at P=2 before new physics is introduced."
        ),
    ),
    PhysicsGate(
        name="low_order_collision_block_recovery",
        category="analytical",
        metric="generated low-order momentum-conserving collision blocks",
        relation="test",
        threshold=None,
        source="standard low-order moment equations and local closure tests",
        rationale=(
            "The active low-order collisional blocks must be reproducible from "
            "the standard momentum-conserving moment equations, up to the "
            "heat-flow basis convention used by the runtime."
        ),
    ),
    PhysicsGate(
        name="observable_map_fixed",
        category="analytical",
        metric="U_parallel = n c0",
        relation="test",
        threshold=None,
        source="closure derivation in the manuscript and fixed-field audits",
        rationale=(
            "The parallel-flow observable is fixed by the Sonine basis and must "
            "not be changed to fit a benchmark."
        ),
    ),
    PhysicsGate(
        name="intrinsic_ambipolarity_symmetric_limit",
        category="analytical",
        metric="symmetric-limit ambipolar structure preserved",
        relation="test",
        threshold=None,
        source=(
            "Sugama-Nishimura finite-order moment-equation requirements and "
            "tests/test_physics_gates.py"
        ),
        rationale=(
            "At every finite truncation, the projected closure must preserve the "
            "intrinsic ambipolar-diffusion structure in symmetric limits."
        ),
    ),
    PhysicsGate(
        name="spitzer_inverse_collisionality_limit",
        category="analytical",
        metric="constant-field D33_spitzer proportional to 1/nu_hat",
        relation="test",
        threshold=None,
        source="tests/test_physics_gates.py",
        rationale=(
            "In the constant-field limit the drift source vanishes and the "
            "remaining parallel-conductivity branch should reduce to the "
            "Spitzer-like inverse-collisionality normalization used by NTX."
        ),
    ),
    PhysicsGate(
        name="constant_field_radial_electric_field_invariance",
        category="analytical",
        metric="constant-field transport invariant under er_hat sweep",
        relation="test",
        threshold=None,
        source="tests/test_physics_gates.py",
        rationale=(
            "With constant B on the flux surface, the magnetic-drift source "
            "vanishes. Sweeping the normalized radial electric field must not "
            "create radial transport or alter the parallel-conductivity branch."
        ),
    ),
    PhysicsGate(
        name="boozer_jacobian_identity",
        category="analytical",
        metric="J B^2 = B_zeta + iota B_theta",
        relation="test",
        threshold=None,
        source="tests/test_geometry.py",
        rationale=(
            "The Boozer Jacobian normalization fixes the relation between the "
            "covariant field components, the contravariant field components, "
            "and the drift/source terms consumed by the solver."
        ),
    ),
    PhysicsGate(
        name="operator_parameter_derivative_consistency",
        category="analytical",
        metric="dD_k/dnu_hat and dD_k/depsi_hat match operator autodiff",
        relation="test",
        threshold=None,
        source="tests/test_operators.py",
        rationale=(
            "The implicit-adjoint path differentiates through hand-coded "
            "parameter-derivative blocks, so those blocks must be exactly the "
            "derivatives of the assembled Legendre-space operator with respect "
            "to collisionality and radial-electric-field normalization."
        ),
    ),
    PhysicsGate(
        name="profile_interpolant_parameter_derivative_consistency",
        category="analytical",
        metric="D33 profile sensitivity with respect to electric-field basis parameters",
        relation="test",
        threshold=None,
        source="tests/test_autodiff.py",
        rationale=(
            "Profile-level inverse design and uncertainty propagation depend on "
            "differentiating imported monoenergetic coefficient tables through "
            "the radial-electric-field profile basis, so the interpolated D33 "
            "sensitivity must agree with centered finite differences on a "
            "controlled table."
        ),
    ),
    PhysicsGate(
        name="profile_control_linear_response",
        category="analytical",
        metric="scalar and basis profile controls preserve identity and linear response",
        relation="test",
        threshold=None,
        source="tests/test_profiles_unit.py",
        rationale=(
            "Profile optimization, sensitivity, and uncertainty workflows rely "
            "on explicit low-dimensional controls being identity maps at zero "
            "control and linear maps in their prescribed response basis."
        ),
    ),
    PhysicsGate(
        name="primitive_profile_force_reconstruction",
        category="analytical",
        metric="A1/A3 from primitive density and temperature profiles",
        relation="test",
        threshold=None,
        source="tests/test_profiles_unit.py",
        rationale=(
            "Profile-level workflows must reconstruct the thermodynamic-force "
            "proxies from primitive density, temperature, charge, and radial "
            "electric-field inputs before those forces are used in particle "
            "flux or bootstrap-current proxy calculations."
        ),
    ),
    PhysicsGate(
        name="charge_symmetric_ambipolar_cancellation",
        category="analytical",
        metric="sum_s Z_s Gamma_s for equal-and-opposite species pair",
        relation="test",
        threshold=None,
        source="tests/test_profiles_unit.py",
        rationale=(
            "The ambipolar radial-electric-field workflow is built on the "
            "charge-weighted particle-flux condition sum_s Z_s Gamma_s = 0, "
            "so equal particle-flux responses with opposite charge must cancel "
            "exactly before any root solve or profile optimization is trusted."
        ),
    ),
    PhysicsGate(
        name="primitive_transport_positivity_floor",
        category="analytical",
        metric="density and temperature stay positive after primitive transport update",
        relation="test",
        threshold=None,
        source="tests/test_profiles_workflows.py",
        rationale=(
            "Primitive profile transport updates act on density and temperature, "
            "which must remain positive thermodynamic state variables even under "
            "large explicit relaxation steps."
        ),
    ),
    PhysicsGate(
        name="vmec_jax_boundary_edge_transfer",
        category="analytical",
        metric="traced boundary edge arrays forwarded to implicit and explicit VMEC solves",
        relation="test",
        threshold=None,
        source="tests/test_vmec_jax_backend.py",
        rationale=(
            "Boundary-to-output derivatives require the traced fixed-boundary "
            "Fourier edge arrays to reach both the implicit residual solve and "
            "the explicit relaxation solve without being replaced by stale "
            "non-differentiated boundary data."
        ),
    ),
    PhysicsGate(
        name="imported_boozer_handedness",
        category="analytical",
        metric="B_zeta + iota B_theta >= 0 after imported Boozer sign mapping",
        relation="test",
        threshold=None,
        source="tests/test_vmec_jax_backend.py",
        rationale=(
            "The in-memory VMEC-to-Boozer path must use the same right-handed "
            "Boozer convention as the file-backed loader before NTX consumes "
            "the Boozer Jacobian and drift source terms."
        ),
    ),
    PhysicsGate(
        name="momentum_conservation_null_mode",
        category="analytical",
        metric="common-flow collisional null mode preserved",
        relation="test",
        threshold=None,
        source="momentum-restoring closure derivation and local closure tests",
        rationale=(
            "The higher-order collisional blocks must conserve total parallel "
            "momentum, so a common-flow null mode remains present."
        ),
    ),
    PhysicsGate(
        name="particle_conservation_invariant",
        category="analytical",
        metric="collisional particle invariant preserved",
        relation="test",
        threshold=None,
        source="linearized collision-operator moment-equation constraints",
        rationale=(
            "The projected collision model must not generate a spurious particle "
            "source at any truncation."
        ),
    ),
    PhysicsGate(
        name="energy_conservation_invariant",
        category="analytical",
        metric="collisional energy invariant preserved",
        relation="test",
        threshold=None,
        source="linearized collision-operator moment-equation constraints",
        rationale=(
            "The collisional blocks must preserve the energy invariant in the "
            "same projected basis used for higher-order closure."
        ),
    ),
    PhysicsGate(
        name="collision_operator_self_adjointness",
        category="analytical",
        metric="weighted collisional form is self-adjoint",
        relation="test",
        threshold=None,
        source="finite-order Laguerre/Sonine Coulomb-operator literature",
        rationale=(
            "The finite-order collisional operator should preserve the "
            "self-adjoint structure underlying Onsager symmetry and the H-theorem."
        ),
    ),
    PhysicsGate(
        name="entropy_production_nonnegative",
        category="analytical",
        metric="symmetric collisional form is positive semidefinite",
        relation="test",
        threshold=None,
        source="Sugama-Horton entropy-production constraints",
        rationale=(
            "The finite-order collision model must not violate the "
            "non-negativity of entropy production."
        ),
    ),
)

__all__ = ["ANALYTICAL_GATES"]
