"""Native bootstrap-current closures built directly on NTX monoenergetic scans."""

from __future__ import annotations

from dataclasses import dataclass

import interpax
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, tree_util

from .neopax import NeopaxMonoenergeticArrays, NeopaxScan, scan_to_neopax_arrays
from .profiles import PrimitiveSpeciesProfile

ELEMENTARY_CHARGE = 1.602176634e-19
EPSILON_0 = 8.8541878128e-12
PROTON_MASS = 1.67262192369e-27
JOULE_PER_EV = 1.602176634e-19


@dataclass(frozen=True)
class BootstrapSpeciesProfile:
    """Species data for native no-momentum bootstrap-current closure."""

    mass_mp: float | Array
    charge_qp: float | Array
    density: Array
    temperature: Array
    A1: Array
    A2: Array
    A3: Array
    name: str | None = None


tree_util.register_dataclass(
    BootstrapSpeciesProfile,
    data_fields=("mass_mp", "charge_qp", "density", "temperature", "A1", "A2", "A3"),
    meta_fields=("name",),
)


@dataclass(frozen=True)
class BootstrapCurrentResult:
    """Native no-momentum bootstrap-current outputs."""

    rho: Array
    er_profile: Array
    Lij: Array
    particle_flux: Array
    heat_flux: Array
    parallel_flow: Array
    current_density: Array
    jdotb: Array


tree_util.register_dataclass(
    BootstrapCurrentResult,
    data_fields=(
        "rho",
        "er_profile",
        "Lij",
        "particle_flux",
        "heat_flux",
        "parallel_flow",
        "current_density",
        "jdotb",
    ),
    meta_fields=(),
)


def build_bootstrap_species_profile(
    rho: Array,
    primitive: PrimitiveSpeciesProfile,
    *,
    mass_mp: float,
    er_profile: Array,
    a_b: float,
    smoothing_strength: float = 0.35,
) -> BootstrapSpeciesProfile:
    """Construct bootstrap-current force profiles from primitive inputs."""

    rho_arr = jnp.asarray(rho)
    density = _broadcast_profile_field(primitive.density, rho_arr)
    temperature = _broadcast_profile_field(primitive.temperature, rho_arr)
    charge = _broadcast_profile_field(primitive.charge, rho_arr)
    er_arr = _broadcast_profile_field(er_profile, rho_arr)
    radius = jnp.maximum(jnp.asarray(a_b) * rho_arr, jnp.asarray(1.0e-8, dtype=rho_arr.dtype))

    density_smooth = _smooth_radial_profile(density, smoothing_strength)
    temperature_smooth = _smooth_radial_profile(temperature, smoothing_strength)
    dndr = _radial_gradient(density_smooth, radius)
    dTdr = _radial_gradient(temperature_smooth, radius)
    safe_density = jnp.maximum(density_smooth, jnp.asarray(1.0e-30, dtype=rho_arr.dtype))
    safe_temperature = jnp.maximum(
        temperature_smooth,
        jnp.asarray(1.0e-30, dtype=rho_arr.dtype),
    )
    a1 = (
        dndr / safe_density
        - 1.5 * dTdr / safe_temperature
        - 1.0e3 * er_arr * charge * ELEMENTARY_CHARGE / (safe_temperature * ELEMENTARY_CHARGE)
    )
    a2 = dTdr / safe_temperature
    a3 = jnp.zeros_like(rho_arr)
    return BootstrapSpeciesProfile(
        mass_mp=jnp.asarray(mass_mp, dtype=rho_arr.dtype),
        charge_qp=primitive.charge,
        density=density_smooth,
        temperature=temperature_smooth,
        A1=a1,
        A2=a2,
        A3=a3,
        name=primitive.name,
    )


def build_bootstrap_species_profiles(
    rho: Array,
    primitives: tuple[PrimitiveSpeciesProfile, ...],
    *,
    mass_mp: tuple[float, ...],
    er_profile: Array,
    a_b: float,
    smoothing_strength: float = 0.35,
) -> tuple[BootstrapSpeciesProfile, ...]:
    """Vectorized primitive-to-bootstrap profile construction helper."""

    if len(primitives) != len(mass_mp):
        raise ValueError("mass_mp must match the number of primitive species")
    return tuple(
        build_bootstrap_species_profile(
            rho,
            primitive,
            mass_mp=mass_value,
            er_profile=er_profile,
            a_b=a_b,
            smoothing_strength=smoothing_strength,
        )
        for primitive, mass_value in zip(primitives, mass_mp, strict=True)
    )


def evaluate_bootstrap_current(
    scan: NeopaxScan,
    species_profiles: tuple[BootstrapSpeciesProfile, ...],
    *,
    a_b: float | None = None,
    er_profile: Array | None = None,
    n_x: int = 64,
    neopax_compat_boundary: bool = False,
) -> BootstrapCurrentResult:
    """Evaluate native no-momentum bootstrap current from an NTX scan."""

    if not species_profiles:
        raise ValueError("species_profiles must not be empty")
    a_b_value = _resolve_a_b(scan, a_b)
    rho = jnp.asarray(scan.rho)
    er_arr = _broadcast_profile_field(
        jnp.zeros_like(rho) if er_profile is None else er_profile,
        rho,
    )
    arrays = scan_to_neopax_arrays(scan, a_b=a_b_value)
    x, weights = _laguerre_grid(n_x, dtype=rho.dtype)
    powers = {
        "L11": x**2,
        "L12": x**3,
        "L22": x**4,
        "L13": x**1.5,
        "L23": x**2.5,
        "L33": x,
    }

    lij_rows = []
    for species_index, species in enumerate(species_profiles):
        lij_species = []
        v_thermal = _thermal_speed(species.mass_mp, species.temperature)
        for radial_index in range(rho.size):
            v_values = jnp.sqrt(x) * v_thermal[radial_index]
            nu_over_v = jnp.asarray(
                [
                    _collisionality(
                        species_profiles,
                        species_index,
                        float(v_value),
                        radial_index,
                    )
                    / float(v_value)
                    for v_value in np.asarray(v_values)
                ],
                dtype=rho.dtype,
            )
            er_over_v = er_arr[radial_index] * 1.0e3 / v_values
            d11, d13, d33 = _interpolate_native_dij(
                arrays,
                radial_index,
                nu_over_v,
                er_over_v,
                radius=jnp.asarray(a_b_value, dtype=rho.dtype) * rho[radial_index],
            )
            lij = _build_lij_matrix(
                species,
                radial_index,
                v_thermal[radial_index],
                d11,
                d13,
                d33,
                weights,
                powers,
            )
            lij_species.append(lij)
        lij_rows.append(jnp.stack(lij_species))

    lij = jnp.stack(lij_rows)
    if neopax_compat_boundary and rho.size > 1:
        lij = lij.at[:, 0, :, :].set(lij[:, 1, :, :])

    gamma_rows = []
    heat_rows = []
    upar_rows = []
    for species_index, species in enumerate(species_profiles):
        gamma_species = []
        heat_species = []
        upar_species = []
        for radial_index in range(rho.size):
            gamma, heat_flux, upar = _evaluate_flux_row(
                species,
                radial_index,
                lij[species_index, radial_index],
            )
            gamma_species.append(gamma)
            heat_species.append(heat_flux)
            upar_species.append(upar)
        gamma_rows.append(jnp.stack(gamma_species))
        heat_rows.append(jnp.stack(heat_species))
        upar_rows.append(jnp.stack(upar_species))

    particle_flux = jnp.stack(gamma_rows)
    heat_flux = jnp.stack(heat_rows)
    parallel_flow = jnp.stack(upar_rows)
    charge_qp = jnp.asarray(
        [jnp.asarray(species.charge_qp) for species in species_profiles],
        dtype=rho.dtype,
    )
    current_density = ELEMENTARY_CHARGE * jnp.sum(charge_qp[:, None] * parallel_flow, axis=0)
    return BootstrapCurrentResult(
        rho=rho,
        er_profile=er_arr,
        Lij=lij,
        particle_flux=particle_flux,
        heat_flux=heat_flux,
        parallel_flow=parallel_flow,
        current_density=current_density,
        jdotb=current_density,
    )


def _resolve_a_b(scan: NeopaxScan, a_b: float | None) -> float:
    if a_b is not None:
        return float(a_b)
    if scan.a_b is not None:
        return float(scan.a_b)
    raise ValueError("a_b must be provided when scan.a_b is not available")


def _laguerre_grid(n_x: int, *, dtype) -> tuple[Array, Array]:
    x, weights = np.polynomial.laguerre.laggauss(int(n_x))
    return jnp.asarray(x, dtype=dtype), jnp.asarray(weights, dtype=dtype)


def _broadcast_profile_field(values, rho: Array) -> Array:
    array = jnp.asarray(values, dtype=jnp.asarray(rho).dtype)
    if array.ndim == 0:
        return jnp.full_like(rho, array)
    if array.shape == rho.shape:
        return array
    raise ValueError("profile field must be scalar or match rho shape")


def _smooth_radial_profile(values: Array, strength: float) -> Array:
    array = jnp.asarray(values)
    if array.size < 3:
        return array
    strength_value = jnp.clip(jnp.asarray(strength, dtype=array.dtype), 0.0, 1.0)
    left = jnp.roll(array, 1)
    right = jnp.roll(array, -1)
    smoothed = (1.0 - strength_value) * array + 0.5 * strength_value * (left + right)
    smoothed = smoothed.at[0].set(array[0])
    smoothed = smoothed.at[-1].set(array[-1])
    return smoothed


def _radial_gradient(values: Array, radius: Array) -> Array:
    array = jnp.asarray(values)
    radius_arr = jnp.asarray(radius)
    if array.size < 2:
        return jnp.zeros_like(array)
    gradient = jnp.zeros_like(array)
    forward = (array[1] - array[0]) / jnp.maximum(
        radius_arr[1] - radius_arr[0],
        jnp.asarray(1.0e-12, dtype=array.dtype),
    )
    backward = (array[-1] - array[-2]) / jnp.maximum(
        radius_arr[-1] - radius_arr[-2],
        jnp.asarray(1.0e-12, dtype=array.dtype),
    )
    gradient = gradient.at[0].set(forward)
    gradient = gradient.at[-1].set(backward)
    if array.size > 2:
        central = (array[2:] - array[:-2]) / jnp.maximum(
            radius_arr[2:] - radius_arr[:-2],
            jnp.asarray(1.0e-12, dtype=array.dtype),
        )
        gradient = gradient.at[1:-1].set(central)
    return gradient


def _species_mass(species: BootstrapSpeciesProfile) -> Array:
    return jnp.asarray(species.mass_mp) * PROTON_MASS


def _species_charge_coulomb(species: BootstrapSpeciesProfile) -> Array:
    return jnp.asarray(species.charge_qp) * ELEMENTARY_CHARGE


def _thermal_speed(mass_mp: float | Array, temperature_eV: Array) -> Array:
    return jnp.sqrt(
        2.0
        * jnp.asarray(temperature_eV)
        * JOULE_PER_EV
        / (jnp.asarray(mass_mp) * PROTON_MASS)
    )


def _electron_reference(
    species_profiles: tuple[BootstrapSpeciesProfile, ...],
) -> BootstrapSpeciesProfile:
    return min(species_profiles, key=lambda species: float(jnp.asarray(species.charge_qp)))


def _coulomb_logarithm(
    species_profiles: tuple[BootstrapSpeciesProfile, ...],
    radial_index: int,
) -> Array:
    electron = _electron_reference(species_profiles)
    temperature = jnp.maximum(
        jnp.asarray(electron.temperature)[radial_index],
        jnp.asarray(1.0e-12, dtype=jnp.asarray(electron.temperature).dtype),
    )
    density = jnp.maximum(
        jnp.asarray(electron.density)[radial_index],
        jnp.asarray(1.0e6, dtype=jnp.asarray(electron.density).dtype),
    )
    return 32.2 + 1.15 * jnp.log10(temperature**2 / density)


def _gamma_ab(
    species_profiles: tuple[BootstrapSpeciesProfile, ...],
    species_a: int,
    species_b: int,
    radial_index: int,
) -> Array:
    charge_a = _species_charge_coulomb(species_profiles[species_a])
    charge_b = _species_charge_coulomb(species_profiles[species_b])
    mass_a = _species_mass(species_profiles[species_a])
    lnlambda = _coulomb_logarithm(species_profiles, radial_index)
    return charge_a**2 * charge_b**2 * lnlambda / (4.0 * jnp.pi * EPSILON_0**2 * mass_a**2)


def _chandrasekhar(x: Array) -> Array:
    x_arr = jnp.asarray(x)
    erf = jax.scipy.special.erf(x_arr)
    prefactor = 2.0 * x_arr / jnp.sqrt(jnp.pi)
    numerator = erf - prefactor * jnp.exp(-(x_arr**2))
    denominator = 2.0 * jnp.maximum(x_arr**2, jnp.asarray(1.0e-30, dtype=x_arr.dtype))
    return numerator / denominator


def _nuD_ab(
    species_profiles: tuple[BootstrapSpeciesProfile, ...],
    species_a: int,
    species_b: int,
    velocity: float,
    radial_index: int,
) -> Array:
    density_b = jnp.asarray(species_profiles[species_b].density)[radial_index]
    vtb = _thermal_speed(
        species_profiles[species_b].mass_mp,
        species_profiles[species_b].temperature,
    )[radial_index]
    velocity_value = jnp.asarray(velocity, dtype=jnp.asarray(density_b).dtype)
    prefactor = (
        _gamma_ab(species_profiles, species_a, species_b, radial_index)
        * density_b
        / velocity_value**3
    )
    erf_part = jax.scipy.special.erf(velocity_value / vtb) - _chandrasekhar(velocity_value / vtb)
    return prefactor * erf_part


def _collisionality(
    species_profiles: tuple[BootstrapSpeciesProfile, ...],
    species_a: int,
    velocity: float,
    radial_index: int,
) -> Array:
    return jnp.sum(
        jnp.asarray(
            [
                _nuD_ab(species_profiles, species_a, species_b, velocity, radial_index)
                for species_b in range(len(species_profiles))
            ]
        )
    )


def _interpolate_native_dij(
    arrays: NeopaxMonoenergeticArrays,
    radial_index: int,
    nu_over_v: Array,
    er_over_v: Array,
    *,
    radius: Array,
) -> tuple[Array, Array, Array]:
    nu_axis = jnp.asarray(arrays.nu_log)
    nu_values = jnp.maximum(jnp.asarray(nu_over_v), jnp.asarray(1.0e-30, dtype=nu_axis.dtype))
    if arrays.rho.size < 3:
        sampled = jnp.asarray(
            [
                _interpolate_radial_slice(
                    arrays,
                    radial_index,
                    float(jnp.log10(nu_value)),
                    float(
                        jnp.log10(
                            jnp.maximum(
                                1.0e-8,
                                jnp.abs(er_value)
                                / jnp.maximum(
                                    jnp.asarray(radius, dtype=nu_axis.dtype),
                                    jnp.asarray(1.0e-8, dtype=nu_axis.dtype),
                                ),
                            )
                        )
                    ),
                )
                for nu_value, er_value in zip(
                    np.asarray(nu_values),
                    np.asarray(er_over_v),
                    strict=True,
                )
            ],
            dtype=nu_axis.dtype,
        )
        return (
            -10.0 ** sampled[:, 0],
            -sampled[:, 1],
            -sampled[:, 2] / nu_values,
        )
    radius_value = jnp.asarray(radius, dtype=nu_axis.dtype)
    d11_values = jnp.asarray(
        [
            _interpolate_database_entry(
                arrays,
                float(radius_value),
                float(nu_value),
                float(er_value),
            )
            for nu_value, er_value in zip(
                np.asarray(nu_values),
                np.asarray(er_over_v),
                strict=True,
            )
        ],
        dtype=nu_axis.dtype,
    )
    d11 = -10.0 ** d11_values[:, 0]
    d13 = -d11_values[:, 1]
    d33 = -d11_values[:, 2] / nu_values
    return d11, d13, d33


def _interpolate_database_entry(
    arrays: NeopaxMonoenergeticArrays,
    radius: float,
    nu_value: float,
    er_over_v: float,
) -> Array:
    grid_nu = float(jnp.log10(max(1.0e-12, nu_value)))
    if radius <= float(1.0e-3 * arrays.a_b):
        grid_er = float(jnp.log10(jnp.asarray(1.0e-8, dtype=arrays.rho.dtype)))
    else:
        grid_er = float(jnp.log10(max(1.0e-8, abs(er_over_v / radius))))
    if radius < float(arrays.a_b * arrays.rho[1]):
        return _interpolate_small_radius(arrays, radius, grid_nu, grid_er)
    if radius < float(arrays.a_b * arrays.rho[-2]):
        return _interpolate_mid_radius(arrays, radius, grid_nu, grid_er)
    return _interpolate_large_radius(arrays, radius, grid_nu, grid_er)


def _interpolate_small_radius(
    arrays: NeopaxMonoenergeticArrays,
    radius: float,
    log_nu: float,
    log_er: float,
) -> Array:
    xr2 = radius**2
    xr3 = radius**3
    radii = np.asarray(arrays.rho[:3] * arrays.a_b, dtype=float)
    r12, r22, r32 = radii**2
    r13, r23, r33 = radii**3
    rows = [
        _interpolate_radial_slice(arrays, index, log_nu, log_er)
        for index in range(3)
    ]
    rows_array = np.stack(rows, axis=0)
    result = []
    for channel_values in rows_array.T:
        v0, v1, v2 = channel_values
        ha1 = ((v2 - v1) / (r33 - r23) - (v2 - v0) / (r33 - r13)) / (
            (r32 - r22) / (r33 - r23) - (r32 - r12) / (r33 - r13)
        )
        hb1 = ((v2 - v1) / (r32 - r22) - (v2 - v0) / (r32 - r12)) / (
            (r33 - r23) / (r32 - r22) - (r33 - r13) / (r32 - r12)
        )
        hg1 = v0 - r12 * ha1 - r13 * hb1
        result.append(hg1 + xr2 * ha1 + xr3 * hb1)
    return jnp.asarray(result, dtype=arrays.rho.dtype)


def _interpolate_large_radius(
    arrays: NeopaxMonoenergeticArrays,
    radius: float,
    log_nu: float,
    log_er: float,
) -> Array:
    radii = np.asarray(arrays.rho[-3:] * arrays.a_b, dtype=float)
    hr0 = (radius - radii[1]) * (radius - radii[2]) / (
        (radii[0] - radii[1]) * (radii[0] - radii[2])
    )
    hr1 = (radius - radii[0]) * (radius - radii[2]) / (
        (radii[1] - radii[0]) * (radii[1] - radii[2])
    )
    hr2 = (radius - radii[0]) * (radius - radii[1]) / (
        (radii[2] - radii[0]) * (radii[2] - radii[1])
    )
    rows = [
        _interpolate_radial_slice(arrays, index, log_nu, log_er)
        for index in range(arrays.rho.size - 3, arrays.rho.size)
    ]
    rows_array = np.stack(rows, axis=0)
    return jnp.asarray(
        hr0 * rows_array[0] + hr1 * rows_array[1] + hr2 * rows_array[2],
        dtype=arrays.rho.dtype,
    )


def _interpolate_mid_radius(
    arrays: NeopaxMonoenergeticArrays,
    radius: float,
    log_nu: float,
    log_er: float,
) -> Array:
    radial_grid = np.asarray(arrays.rho * arrays.a_b, dtype=float)
    interior = radial_grid[1:-1]
    index = int(np.argmax(np.where(radius - interior <= 0.0, radius - interior, -np.inf))) + 1
    idx0 = index - 2
    idx1 = index - 1
    idx2 = index
    idx3 = index + 1
    r0 = radial_grid[idx0]
    r1 = radial_grid[idx1]
    r2 = radial_grid[idx2]
    r3 = radial_grid[idx3]
    hr0 = (radius - r1) * (radius - r2) * (radius - r3) / ((r0 - r1) * (r0 - r2) * (r0 - r3))
    hr1 = (radius - r0) * (radius - r2) * (radius - r3) / ((r1 - r0) * (r1 - r2) * (r1 - r3))
    hr2 = (radius - r0) * (radius - r1) * (radius - r3) / ((r2 - r0) * (r2 - r1) * (r2 - r3))
    hr3 = (radius - r0) * (radius - r1) * (radius - r2) / ((r3 - r0) * (r3 - r1) * (r3 - r2))
    rows = [
        _interpolate_radial_slice(arrays, index_value, log_nu, log_er)
        for index_value in (idx0, idx1, idx2, idx3)
    ]
    rows_array = np.stack(rows, axis=0)
    return jnp.asarray(
        hr0 * rows_array[0] + hr1 * rows_array[1] + hr2 * rows_array[2] + hr3 * rows_array[3],
        dtype=arrays.rho.dtype,
    )


def _interpolate_d11_native(
    nu_axis: Array,
    er_axis: Array,
    d11_log_slice: Array,
    log_nu: Array,
    log_er: Array,
) -> Array:
    er_floor = jnp.log10(jnp.asarray(1.0e-8, dtype=nu_axis.dtype))
    if log_er <= er_floor:
        if log_nu < nu_axis[0]:
            return d11_log_slice[0, 0] + (nu_axis[0] - log_nu)
        if log_nu > nu_axis[-1]:
            return d11_log_slice[-1, 0] + (log_nu - nu_axis[-1])
        return interpax.Interpolator1D(nu_axis, d11_log_slice[:, 0], extrap=False)(log_nu)
    if log_nu < nu_axis[0]:
        er_h = (nu_axis[0] - log_nu) / 3.0 + log_er
        return jnp.where(
            er_h < er_axis[-1],
            interpax.Interpolator1D(er_axis, d11_log_slice[0, :], extrap=False)(er_h),
            jnp.asarray(-12.0, dtype=nu_axis.dtype),
        )
    if log_nu > nu_axis[-1]:
        return interpax.Interpolator2D(nu_axis, er_axis, d11_log_slice, extrap=True)(log_nu, log_er)
    if log_nu <= nu_axis[1]:
        return _lagrange_interpolate_three(
            nu_axis[:3],
            _sample_d11_along_er(er_axis, d11_log_slice[:3, :], log_er),
            log_nu,
        )
    if log_nu >= nu_axis[-2]:
        return _lagrange_interpolate_three(
            nu_axis[-3:],
            _sample_d11_along_er(er_axis, d11_log_slice[-3:, :], log_er),
            log_nu,
        )
    return _hermite_interpolate_four(
        nu_axis,
        er_axis,
        d11_log_slice,
        log_nu,
        log_er,
    )


def _interpolate_radial_slice(
    arrays: NeopaxMonoenergeticArrays,
    radial_index: int,
    log_nu: float,
    log_er: float,
) -> Array:
    nu_axis = jnp.asarray(arrays.nu_log)
    er_axis = jnp.asarray(arrays.Er_list[radial_index])
    d11_interp = interpax.Interpolator2D(
        nu_axis,
        er_axis,
        jnp.asarray(arrays.D11_log[radial_index]),
        extrap=True,
    )
    d13_interp = interpax.Interpolator2D(
        nu_axis,
        er_axis,
        jnp.asarray(arrays.D13[radial_index]),
        extrap=True,
    )
    d33_interp = interpax.Interpolator2D(
        nu_axis,
        er_axis,
        jnp.asarray(arrays.D33[radial_index]),
        extrap=True,
    )
    return jnp.asarray(
        [
            d11_interp(log_nu, log_er),
            d13_interp(log_nu, log_er),
            d33_interp(log_nu, log_er),
        ],
        dtype=nu_axis.dtype,
    )


def _sample_d11_along_er(er_axis: Array, values: Array, log_er: Array) -> Array:
    low = values[:, 0]
    finite = jax.vmap(
        lambda row: interpax.Interpolator1D(er_axis, row, extrap=False)(log_er)
    )(values)
    high = jnp.full_like(low, -12.0)
    return jnp.where(
        log_er <= er_axis[0],
        low,
        jnp.where(log_er <= er_axis[-1], finite, high),
    )


def _lagrange_interpolate_three(nu_points: Array, values: Array, log_nu: Array) -> Array:
    nu0, nu1, nu2 = nu_points
    d0, d1, d2 = values
    h0 = (log_nu - nu1) * (log_nu - nu2) / ((nu0 - nu1) * (nu0 - nu2))
    h1 = (log_nu - nu0) * (log_nu - nu2) / ((nu1 - nu0) * (nu1 - nu2))
    h2 = (log_nu - nu0) * (log_nu - nu1) / ((nu2 - nu0) * (nu2 - nu1))
    return h0 * d0 + h1 * d1 + h2 * d2


def _hermite_interpolate_four(
    nu_axis: Array,
    er_axis: Array,
    d11_log_slice: Array,
    log_nu: Array,
    log_er: Array,
) -> Array:
    arr = log_nu - nu_axis[1:-1]
    index = int(jnp.argmax(jnp.where(arr <= 0.0, arr, -jnp.inf))) + 1
    idx0 = index - 1
    idx1 = index - 2
    idx2 = index
    idx3 = index + 1
    nu0 = float(nu_axis[idx0])
    nu1 = float(nu_axis[idx1])
    nu2 = float(nu_axis[idx2])
    nu3 = float(nu_axis[idx3])
    values = _sample_d11_along_er(
        er_axis,
        jnp.asarray(
            [
                d11_log_slice[idx0, :],
                d11_log_slice[idx1, :],
                d11_log_slice[idx2, :],
                d11_log_slice[idx3, :],
            ]
        ),
        log_er,
    )
    d0 = float(values[0])
    d1 = float(values[1])
    d2 = float(values[2])
    d3 = float(values[3])
    if d2 <= -12.0:
        return jnp.asarray(-12.0, dtype=nu_axis.dtype)

    dhl0 = (nu1 - nu2) / ((nu0 - nu1) * (nu0 - nu2))
    dhl1 = (2.0 * nu1 - nu0 - nu2) / ((nu1 - nu0) * (nu1 - nu2))
    dhl2 = (nu1 - nu0) / ((nu2 - nu0) * (nu2 - nu1))
    dhu0 = (nu2 - nu3) / ((nu1 - nu2) * (nu1 - nu3))
    dhu1 = (2.0 * nu2 - nu1 - nu3) / ((nu2 - nu1) * (nu2 - nu3))
    dhu2 = (nu2 - nu1) / ((nu3 - nu1) * (nu3 - nu2))
    dg11l = dhl0 * d0 + dhl1 * d1 + dhl2 * d2
    dg11u = dhu0 * d1 + dhu1 * d2 + dhu2 * d3
    dg11_new = (
        jnp.asarray([0.2 * dg11l, 0.2 * dg11u], dtype=nu_axis.dtype)
        if dg11l * dg11u <= 0.0
        else jnp.asarray([dg11l, dg11u], dtype=nu_axis.dtype)
    )
    dxnu21 = nu2 - nu1
    xnun = (float(log_nu) - nu1) / dxnu21
    ha1 = 3.0 * (d2 - d1) - (2.0 * dg11_new[0] + dg11_new[1]) * dxnu21
    hb1 = -2.0 * (d2 - d1) + (dg11_new[0] + dg11_new[1]) * dxnu21
    return jnp.asarray(
        d1 + xnun * (dg11_new[0] * dxnu21 + xnun * (ha1 + xnun * hb1)),
        dtype=nu_axis.dtype,
    )


def _build_lij_matrix(
    species: BootstrapSpeciesProfile,
    radial_index: int,
    v_thermal: Array,
    d11: Array,
    d13: Array,
    d33: Array,
    weights: Array,
    powers: dict[str, Array],
) -> Array:
    mass = _species_mass(species)
    charge = _species_charge_coulomb(species)
    l11_fac = -1.0 / jnp.sqrt(jnp.pi) * (mass / charge) ** 2 * v_thermal**3
    l13_fac = -1.0 / jnp.sqrt(jnp.pi) * (mass / charge) * v_thermal**2
    l33_fac = -1.0 / jnp.sqrt(jnp.pi) * v_thermal
    lij = jnp.zeros((3, 3), dtype=d11.dtype)
    lij = lij.at[0, 0].set(l11_fac * jnp.sum(powers["L11"] * weights * d11))
    lij = lij.at[0, 1].set(l11_fac * jnp.sum(powers["L12"] * weights * d11))
    lij = lij.at[1, 0].set(lij[0, 1])
    lij = lij.at[1, 1].set(l11_fac * jnp.sum(powers["L22"] * weights * d11))
    lij = lij.at[0, 2].set(l13_fac * jnp.sum(powers["L13"] * weights * d13))
    lij = lij.at[1, 2].set(l13_fac * jnp.sum(powers["L23"] * weights * d13))
    lij = lij.at[2, 0].set(-lij[0, 2])
    lij = lij.at[2, 1].set(-lij[1, 2])
    lij = lij.at[2, 2].set(l33_fac * jnp.sum(powers["L33"] * weights * d33))
    return lij


def _evaluate_flux_row(
    species: BootstrapSpeciesProfile,
    radial_index: int,
    lij: Array,
) -> tuple[Array, Array, Array]:
    density = jnp.asarray(species.density)[radial_index]
    temperature = jnp.asarray(species.temperature)[radial_index]
    a1 = jnp.asarray(species.A1)[radial_index]
    a2 = jnp.asarray(species.A2)[radial_index]
    a3 = jnp.asarray(species.A3)[radial_index]
    gamma = -density * (lij[0, 0] * a1 + lij[0, 1] * a2 + lij[0, 2] * a3)
    heat_flux = -temperature * density * (lij[1, 0] * a1 + lij[1, 1] * a2 + lij[1, 2] * a3)
    upar = -density * (lij[2, 0] * a1 + lij[2, 1] * a2 + lij[2, 2] * a3)
    return gamma, heat_flux, upar
