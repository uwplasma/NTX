"""Transport-loop helpers for profile-grade NTX workflows."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ._profiles_ambipolar_types import (
    AmbipolarProfileResult,
)
from ._profiles_eval import (
    build_species_profiles_from_primitives,
    solve_ambipolar_er_profile,
)
from ._profiles_radial import _broadcast_profile_field
from ._profiles_species_types import (
    MonoenergeticSpeciesProfile,
    PrimitiveSpeciesProfile,
)
from ._profiles_transport_closure import (
    advance_primitive_profile_transport,
    advance_profile_transport,
    primitive_profile_transport_loss,
    profile_transport_loss,
)
from ._profiles_transport_terms import (
    _broadcast_species_transport_field as _broadcast_species_transport_field,
)
from ._profiles_transport_terms import _scaled_transport_closure
from ._profiles_transport_types import (
    PrimitiveProfileTransportIterationResult,
    ProfileTransportClosureSpec,
    ProfileTransportIterationResult,
)
from .neopax import NeopaxScan


def solve_profile_transport_loop(
    scan: NeopaxScan,
    species_profiles: tuple[MonoenergeticSpeciesProfile, ...],
    closure_spec: ProfileTransportClosureSpec,
    *,
    iterations: int = 8,
    er_initial: Array | None = None,
    solve_steps: int = 16,
    damping: float = 0.8,
    smoothing_strength: float = 0.0,
) -> ProfileTransportIterationResult:
    """Iterate a simple self-consistent profile transport closure."""

    species_state = species_profiles
    er_seed = er_initial
    profile_history: list[AmbipolarProfileResult] = []
    loss_history: list[Array] = []
    a1_history: list[Array] = []
    a3_history: list[Array] = []
    best_profile: AmbipolarProfileResult | None = None
    best_loss: Array | None = None

    for _ in range(iterations):
        profile = solve_ambipolar_er_profile(
            scan,
            species_state,
            er_initial=er_seed,
            steps=solve_steps,
            damping=damping,
            smoothing_strength=smoothing_strength,
        )
        profile_history.append(profile)
        current_loss = profile_transport_loss(profile, closure_spec)
        loss_history.append(current_loss)
        a1_history.append(jnp.stack([jnp.asarray(species.A1) for species in species_state]))
        a3_history.append(jnp.stack([jnp.asarray(species.A3) for species in species_state]))
        if best_profile is None or best_loss is None or bool(current_loss < best_loss):
            best_profile = profile
            best_loss = current_loss

        next_species_state = species_state
        next_er_seed = profile.er_profile
        accepted = False
        for step_index in range(5):
            factor = jnp.asarray(0.5**step_index, dtype=jnp.asarray(scan.rho).dtype)
            scaled_closure = _scaled_transport_closure(closure_spec, factor)
            candidate_species = advance_profile_transport(species_state, profile, scaled_closure)
            candidate_profile = solve_ambipolar_er_profile(
                scan,
                candidate_species,
                er_initial=profile.er_profile,
                steps=solve_steps,
                damping=damping,
                smoothing_strength=smoothing_strength,
            )
            candidate_loss = profile_transport_loss(candidate_profile, closure_spec)
            if bool(candidate_loss <= current_loss + 1.0e-12):
                next_species_state = candidate_species
                next_er_seed = candidate_profile.er_profile
                accepted = True
                if best_loss is None or bool(candidate_loss < best_loss):
                    best_profile = candidate_profile
                    best_loss = candidate_loss
                break
        if not accepted:
            next_er_seed = profile.er_profile
        species_state = next_species_state
        er_seed = next_er_seed

    loss_array = jnp.stack(loss_history)
    best_index = int(jnp.argmin(loss_array))
    return ProfileTransportIterationResult(
        er_profile_history=jnp.stack([profile.er_profile for profile in profile_history]),
        ambipolar_residual_history=jnp.stack(
            [profile.ambipolar_residual for profile in profile_history]
        ),
        bootstrap_current_response_history=jnp.stack(
            [profile.bootstrap_current_response for profile in profile_history]
        ),
        transport_loss_history=loss_array,
        species_a1_history=jnp.stack(a1_history),
        species_a3_history=jnp.stack(a3_history),
        best_profile=best_profile if best_profile is not None else profile_history[best_index],
    )


def solve_primitive_profile_transport_loop(
    scan: NeopaxScan,
    primitive_profiles: tuple[PrimitiveSpeciesProfile, ...],
    closure_spec: ProfileTransportClosureSpec,
    *,
    iterations: int = 8,
    er_initial: Array | None = None,
    solve_steps: int = 16,
    damping: float = 0.8,
    smoothing_strength: float = 0.0,
) -> PrimitiveProfileTransportIterationResult:
    """Iterate a primitive density/temperature transport closure."""

    primitive_state = primitive_profiles
    rho = jnp.asarray(scan.rho)
    er_seed = (
        0.5 * (jnp.min(jnp.asarray(scan.Er), axis=1) + jnp.max(jnp.asarray(scan.Er), axis=1))
        if er_initial is None
        else _broadcast_profile_field(er_initial, rho)
    )
    profile_history: list[AmbipolarProfileResult] = []
    density_history: list[Array] = []
    temperature_history: list[Array] = []
    loss_history: list[Array] = []
    best_profile: AmbipolarProfileResult | None = None
    best_loss: Array | None = None

    for _ in range(iterations):
        species_profiles = build_species_profiles_from_primitives(
            rho,
            primitive_state,
            er_profile=er_seed,
        )
        profile = solve_ambipolar_er_profile(
            scan,
            species_profiles,
            er_initial=er_seed,
            steps=solve_steps,
            damping=damping,
            smoothing_strength=smoothing_strength,
        )
        profile_history.append(profile)
        density_history.append(
            jnp.stack(
                [
                    _broadcast_profile_field(primitive.density, rho)
                    for primitive in primitive_state
                ]
            )
        )
        temperature_history.append(
            jnp.stack(
                [
                    _broadcast_profile_field(primitive.temperature, rho)
                    for primitive in primitive_state
                ]
            )
        )
        current_loss = primitive_profile_transport_loss(profile, primitive_state, closure_spec)
        loss_history.append(current_loss)
        if best_profile is None or best_loss is None or bool(current_loss < best_loss):
            best_profile = profile
            best_loss = current_loss

        next_primitive_state = primitive_state
        next_er_seed = profile.er_profile
        accepted = False
        for step_index in range(5):
            factor = jnp.asarray(0.5**step_index, dtype=rho.dtype)
            scaled_closure = _scaled_transport_closure(closure_spec, factor)
            candidate_primitive = advance_primitive_profile_transport(
                primitive_state,
                profile,
                scaled_closure,
            )
            candidate_species = build_species_profiles_from_primitives(
                rho,
                candidate_primitive,
                er_profile=profile.er_profile,
            )
            candidate_profile = solve_ambipolar_er_profile(
                scan,
                candidate_species,
                er_initial=profile.er_profile,
                steps=solve_steps,
                damping=damping,
                smoothing_strength=smoothing_strength,
            )
            candidate_loss = primitive_profile_transport_loss(
                candidate_profile,
                candidate_primitive,
                closure_spec,
            )
            if bool(candidate_loss <= current_loss + 1.0e-12):
                next_primitive_state = candidate_primitive
                next_er_seed = candidate_profile.er_profile
                accepted = True
                if best_loss is None or bool(candidate_loss < best_loss):
                    best_profile = candidate_profile
                    best_loss = candidate_loss
                break
        if not accepted:
            next_er_seed = profile.er_profile
        primitive_state = next_primitive_state
        er_seed = next_er_seed

    loss_array = jnp.stack(loss_history)
    best_index = int(jnp.argmin(loss_array))
    return PrimitiveProfileTransportIterationResult(
        er_profile_history=jnp.stack([profile.er_profile for profile in profile_history]),
        ambipolar_residual_history=jnp.stack(
            [profile.ambipolar_residual for profile in profile_history]
        ),
        bootstrap_current_response_history=jnp.stack(
            [profile.bootstrap_current_response for profile in profile_history]
        ),
        transport_loss_history=loss_array,
        species_density_history=jnp.stack(density_history),
        species_temperature_history=jnp.stack(temperature_history),
        best_profile=best_profile if best_profile is not None else profile_history[best_index],
    )
