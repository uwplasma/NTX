from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp

from ntx import (
    GridSpec,
    MonoenergeticSpeciesProfile,
    build_ntx_neopax_scan_from_surfaces,
    example_surface,
)


def example_scan():
    base = example_surface()
    rho = jnp.asarray([0.25, 0.5, 0.75])
    surfaces = tuple(
        replace(base, b_cos=base.b_cos.at[1].set(base.b_cos[1] * (1.0 + 0.15 * float(r))))
        for r in rho
    )
    nu_v = jnp.asarray([3.0e-4, 1.0e-3, 3.0e-3])
    er_axis = jnp.asarray([-2.0e-3, -5.0e-4, 0.0, 5.0e-4, 2.0e-3])
    er = jnp.tile(er_axis[None, :], (rho.size, 1))
    return build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=er,
        Er=er,
        drds=jnp.ones_like(rho),
        grid=GridSpec(5, 5, 4),
        source_name="profile_test",
    )


def species_profiles():
    return (
        MonoenergeticSpeciesProfile(
            charge=-1.0,
            nu_v=jnp.asarray([4.0e-4, 6.0e-4, 8.0e-4]),
            A1=jnp.asarray([1.1, 1.0, 0.9]),
            A3=jnp.asarray([0.55, 0.5, 0.45]),
            current_weight=-1.0,
            name="electron",
        ),
        MonoenergeticSpeciesProfile(
            charge=1.0,
            nu_v=jnp.asarray([2.0e-3, 2.5e-3, 3.0e-3]),
            A1=jnp.asarray([0.7, 0.8, 0.9]),
            A3=jnp.asarray([0.25, 0.25, 0.25]),
            particle_weight=1.1,
            current_weight=1.0,
            name="ion",
        ),
    )
