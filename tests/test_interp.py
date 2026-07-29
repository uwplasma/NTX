"""The interpolators, pinned against reference implementations and structure.

Three kinds of check, because three different things could go wrong:

* the slope rules must be the published ones -- compared against SciPy's
  ``PchipInterpolator`` and ``Akima1DInterpolator``, and, for the
  mean-of-secants rule NTX used before, against its definition;
* the shape-preserving guarantee must actually hold, since it is the reason
  ``pchip`` was chosen for the coefficient channels;
* the local 2D stencil must agree with whole-grid tables, including at the
  edges, where a naive implementation repeats a node and returns ``NaN``.

SciPy is a development convenience here, not a runtime dependency: the tests
that use it skip if it is absent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ntx._interp import Interpolator1D, interp1d, interp2d_at, slopes

jax.config.update("jax_enable_x64", True)


def _grid(rng, n, lo=0.0, hi=1.0):
    x = np.sort(rng.uniform(lo, hi, n))
    x[0], x[-1] = lo, hi
    return x


@pytest.mark.parametrize("seed", range(6))
def test_pchip_matches_scipy(seed: int) -> None:
    scipy_interp = pytest.importorskip("scipy.interpolate")
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 30))
    x = _grid(rng, n)
    if np.min(np.diff(x)) < 1e-4:
        pytest.skip("degenerate grid")
    f = rng.normal(size=n)
    xq = np.sort(rng.uniform(-0.1, 1.1, 200))
    ref = scipy_interp.PchipInterpolator(x, f, extrapolate=True)(xq)
    got = np.asarray(interp1d(xq, x, f, method="pchip"))
    assert np.allclose(got, ref, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("seed", range(6))
def test_akima_matches_scipy(seed: int) -> None:
    scipy_interp = pytest.importorskip("scipy.interpolate")
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(5, 30))
    x = _grid(rng, n)
    if np.min(np.diff(x)) < 1e-4:
        pytest.skip("degenerate grid")
    f = rng.normal(size=n)
    xq = np.sort(rng.uniform(-0.1, 1.1, 200))
    ref = scipy_interp.Akima1DInterpolator(x, f, extrapolate=True)(xq)
    got = np.asarray(interp1d(xq, x, f, method="akima"))
    assert np.allclose(got, ref, rtol=1e-11, atol=1e-11)


def test_cubic_is_the_mean_of_adjacent_secants() -> None:
    """`cubic` records the rule NTX used before, so a regression stays possible.

    It was verified against the previous dependency to 4e-15 when the switch
    was made; that package is gone now, so the rule is pinned by its definition
    instead. Interior slopes are the arithmetic
    mean of the adjacent secants and the ends are the one-sided secants.
    """
    rng = np.random.default_rng(7)
    x = _grid(rng, 14)
    f = rng.normal(size=14)
    d = np.diff(f) / np.diff(x)
    expected = np.concatenate([d[:1], 0.5 * (d[:-1] + d[1:]), d[-1:]])
    assert np.allclose(np.asarray(slopes(jnp.asarray(x), jnp.asarray(f), "cubic")),
                       expected, rtol=1e-13, atol=1e-13)


def test_cubic_and_parabolic_agree_on_a_uniform_grid() -> None:
    """Why D13/D31 are unchanged in practice by moving to the parabolic rule.

    On a uniform grid the length-weighted three-point slope reduces to the mean
    of the secants, so the two rules coincide. They differ only where the grid
    is not uniform -- an E_r axis refined near the resonance, say -- and there
    the parabolic rule is the exact one on a quadratic.
    """
    x = jnp.linspace(-1.0, 1.0, 13)
    f = jnp.asarray(np.random.default_rng(0).normal(size=13))
    assert np.allclose(np.asarray(slopes(x, f, "parabolic")),
                       np.asarray(slopes(x, f, "cubic")), atol=1e-13)

    xn = jnp.asarray(np.sort(np.random.default_rng(1).uniform(-1.0, 1.0, 13)))
    quad = 1.0 - 0.7 * xn ** 2 + 0.3 * xn
    exact = -1.4 * xn + 0.3
    par = np.asarray(slopes(xn, quad, "parabolic"))[1:-1]
    cub = np.asarray(slopes(xn, quad, "cubic"))[1:-1]
    assert np.allclose(par, np.asarray(exact)[1:-1], atol=1e-12)
    assert not np.allclose(cub, np.asarray(exact)[1:-1], atol=1e-6)


def test_pchip_does_not_overshoot_monotone_data() -> None:
    """The property the coefficient channels were chosen for.

    A regime knee in log-log is monotone data with an abrupt slope change. An
    unlimited cubic overshoots it; this asserts the monotone rule does not.
    """
    x = np.linspace(0.0, 1.0, 11)
    f = np.concatenate([np.zeros(5), np.ones(6)])   # a step: the hardest case
    xq = np.linspace(0.0, 1.0, 501)
    got = np.asarray(interp1d(xq, x, f, method="pchip"))
    assert got.min() >= -1e-12, "undershot below the data"
    assert got.max() <= 1.0 + 1e-12, "overshot above the data"
    # And the unlimited rule really does overshoot, so the test has teeth.
    loose = np.asarray(interp1d(xq, x, f, method="cubic"))
    assert loose.max() > 1.0 + 1e-6 or loose.min() < -1e-6


def test_parabolic_keeps_a_smooth_extremum() -> None:
    """Why D13/D31 do not use the monotone rule.

    A monotone limiter sets the slope to zero at every interior extremum, which
    flattens a genuine smooth maximum. The parabolic rule does not.
    """
    x = np.linspace(-1.0, 1.0, 9)
    f = 1.0 - x ** 2
    # Interior only: the parabolic rule is exact on a quadratic where its
    # three-point stencil fits, but the first and last nodes fall back to a
    # one-sided secant, whose slope is the derivative at the interval midpoint
    # rather than at the node. The end intervals are therefore not exact, and
    # asserting otherwise tests the wrong thing.
    xq = np.linspace(x[1], x[-2], 401)
    truth = 1.0 - xq ** 2
    err_par = np.max(np.abs(np.asarray(interp1d(xq, x, f, method="parabolic")) - truth))
    err_pch = np.max(np.abs(np.asarray(interp1d(xq, x, f, method="pchip")) - truth))
    assert err_par < 1e-12, "parabolic should be exact on a quadratic in the interior"
    # The monotone limiter zeroes the slope at the maximum, so it cannot be.
    assert err_pch > 1e-3, "the limiter is expected to flatten the extremum"


def test_interpolants_pass_through_the_nodes() -> None:
    rng = np.random.default_rng(3)
    x = _grid(rng, 12)
    f = rng.normal(size=12)
    for method in ("cubic", "parabolic", "akima", "pchip"):
        got = np.asarray(interp1d(x, x, f, method=method))
        assert np.allclose(got, f, atol=1e-12), method


@pytest.mark.parametrize("method", ["cubic", "parabolic", "akima", "pchip"])
def test_reproduces_a_straight_line_including_outside(method: str) -> None:
    x = np.linspace(0.0, 1.0, 9)
    f = 2.5 * x - 0.75
    xq = np.linspace(-0.5, 1.5, 201)
    got = np.asarray(interp1d(xq, x, f, method=method))
    assert np.allclose(got, 2.5 * xq - 0.75, atol=1e-10)


def test_2d_local_stencil_matches_whole_grid_tables() -> None:
    """Including the edges, where clamped indices would give NaN."""
    from ntx._interp import _interp2d_full

    rng = np.random.default_rng(5)
    x = _grid(rng, 11)
    y = _grid(rng, 9)
    values = rng.normal(size=(11, 9))
    for method in ("cubic", "parabolic", "pchip"):
        for xq in (-0.05, 0.0, 0.013, 0.5, 0.987, 1.0, 1.05):
            for yq in (-0.05, 0.0, 0.5, 1.0, 1.05):
                loc = float(interp2d_at(x, y, values, jnp.asarray(xq),
                                        jnp.asarray(yq), method=method))
                full = float(_interp2d_full(jnp.asarray(x), jnp.asarray(y),
                                            jnp.asarray(values), jnp.asarray(xq),
                                            jnp.asarray(yq), method))
                assert np.isfinite(loc), (method, xq, yq)
                assert abs(loc - full) <= 1e-9 * max(abs(full), 1.0), (method, xq, yq)


def test_2d_is_differentiable_and_finite_at_the_corners() -> None:
    rng = np.random.default_rng(9)
    x = _grid(rng, 8)
    y = _grid(rng, 7)
    values = jnp.asarray(rng.normal(size=(8, 7)))

    def loss(v):
        return interp2d_at(x, y, v, jnp.asarray(0.0), jnp.asarray(1.0),
                           method="pchip")

    g = jax.grad(loss)(values)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_interpolator1d_is_a_pytree_and_differentiable_in_the_query() -> None:
    """NTX takes `jax.grad` of the b00 interpolant to build b0prime."""
    # Assert convergence rather than a hand-picked tolerance: refining the
    # grid must reduce the derivative error, and at a rate consistent with a
    # cubic interpolant. A fixed threshold here would only encode whatever the
    # error happened to be on the day it was written.
    q = jnp.linspace(0.05, 0.95, 40)
    truth = 3.0 * jnp.cos(3.0 * q)
    errs = []
    for n in (12, 24, 48):
        x = jnp.linspace(0.0, 1.0, n)
        itp = Interpolator1D(x, jnp.sin(3.0 * x), method="akima")
        d = jax.vmap(jax.grad(lambda v: itp(jnp.atleast_1d(v))[0]))(q)
        errs.append(float(jnp.max(jnp.abs(d - truth))))
    assert errs[1] < errs[0] / 3.0, f"no convergence under refinement: {errs}"
    assert errs[2] < errs[1] / 3.0, f"no convergence under refinement: {errs}"
    # Relative to the scale of the derivative itself (|f'| <= 3 here), not an
    # absolute number: 2.6e-3 on a quantity of order 3 is under a part in a
    # thousand, and the convergence above is the property that matters.
    assert errs[2] < 1e-3 * float(jnp.max(jnp.abs(truth))), (
        f"derivative still inaccurate at n=48: {errs}")

    x = jnp.linspace(0.0, 1.0, 12)
    itp = Interpolator1D(x, jnp.sin(3.0 * x), method="akima")
    # Survives being passed through a jit boundary as a pytree.
    assert float(jax.jit(lambda o: o(jnp.array([0.4]))[0])(itp)) == pytest.approx(
        float(itp(jnp.array([0.4]))[0])
    )


def test_slopes_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        slopes(jnp.linspace(0, 1, 5), jnp.zeros(5), method="quintic")
