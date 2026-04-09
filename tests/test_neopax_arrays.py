from __future__ import annotations

import jax
import jax.numpy as jnp

from ntx import (
    GridSpec,
    build_ntx_neopax_scan,
    build_ntx_neopax_scan_from_surfaces,
    example_surface,
    scan_to_neopax_arrays,
)


def test_build_ntx_neopax_scan_from_surfaces_matches_callback_builder():
    surfaces = (example_surface(), example_surface())
    rho = jnp.asarray([0.25, 0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    es = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    er = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    drds = jnp.asarray([1.0, 1.5])
    grid = GridSpec(5, 5, 4)

    explicit = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=grid,
        source_name="explicit",
    )
    callback = build_ntx_neopax_scan(
        lambda rho_value: example_surface(),
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=grid,
        source_name="callback",
    )

    assert jnp.allclose(explicit.D11, callback.D11)
    assert jnp.allclose(explicit.D13, callback.D13)
    assert jnp.allclose(explicit.D33, callback.D33)


def test_scan_to_neopax_arrays_matches_expected_scalings():
    surfaces = (example_surface(), example_surface())
    rho = jnp.asarray([0.25, 0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    es = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    er = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    drds = jnp.asarray([1.0, 1.5])
    grid = GridSpec(5, 5, 4)

    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=grid,
    )
    mapped = scan_to_neopax_arrays(scan, a_b=1.0)

    assert mapped.D11_log.shape == scan.D11.shape
    assert mapped.D13.shape == scan.D13.shape
    assert mapped.D33.shape == scan.D33.shape
    assert jnp.allclose(mapped.nu_log, jnp.log10(nu_v))
    assert jnp.allclose(mapped.D13, scan.D13 * drds[:, None, None])
    assert jnp.allclose(mapped.D33, scan.D33 * nu_v[None, :, None])


def test_scan_to_neopax_arrays_is_differentiable_in_es():
    surfaces = (example_surface(), example_surface())
    rho = jnp.asarray([0.25, 0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    er = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    drds = jnp.asarray([1.0, 1.5])
    grid = GridSpec(5, 5, 4)

    def objective(flat_es):
        es = flat_es.reshape(2, 2)
        scan = build_ntx_neopax_scan_from_surfaces(
            surfaces,
            rho=rho,
            nu_v=nu_v,
            Es=es,
            Er=er,
            drds=drds,
            grid=grid,
        )
        mapped = scan_to_neopax_arrays(scan, a_b=1.0)
        return jnp.sum(mapped.D13) + jnp.sum(mapped.D11_log)

    grad = jax.grad(objective)(jnp.asarray([0.0, 1.0e-3, 0.0, 2.0e-3]))
    assert grad.shape == (4,)
    assert jnp.all(jnp.isfinite(grad))
