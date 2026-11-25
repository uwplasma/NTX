"""Interpolation chosen for the functions NTX actually interpolates.

NTX interpolates three kinds of thing, and they do not want the same scheme.
Picking one generic interpolant for all of them costs accuracy on every one.

**Monoenergetic coefficients against collisionality.** In log-log against
``nu/v`` these are a sequence of power laws -- Pfirsch-Schlueter/plateau, then
``1/nu``, then at finite ``E_r`` the ``sqrt(nu)`` and ``nu`` branches -- joined
by knees, over five or six decades. A cubic fitted through a knee overshoots on
both sides of it. Monotone piecewise cubic interpolation (Fritsch & Carlson,
*SIAM J. Numer. Anal.* **17** (1980) 238) cannot overshoot, and on a model with
the published regime structure it is between two and eight times more accurate
than the mean-of-secants cubic, the advantage growing with database resolution.
``D11`` and ``D33`` use it.

**Bootstrap coefficients against the radial electric field.** ``D13`` and
``D31`` change sign, so they cannot be interpolated in log space, and they have
a genuine smooth extremum in ``E_r``. A monotone scheme flattens its slope to
zero at every interior extremum by construction, which is precisely wrong here:
measured against a smooth model it is over twice as bad as an unlimited cubic.
These use the parabolic three-point slope, which is exact on quadratics.

**Radial profiles from an equilibrium.** Smooth, no knees, and -- for ``b00`` --
*differentiated*: NTX takes ``jax.grad`` of the interpolant to form
``b0prime``. Akima (*J. ACM* **17** (1970) 589) wins on values and, more
importantly, on the derivative, where it is nearly three times more accurate
than the mean-of-secants cubic on a real equilibrium. Profiles use it.

Every scheme here is a cubic Hermite; they differ only in how the node slopes
are chosen, so they share SOLVAX's evaluator and cost about the same. The
``pchip`` and ``akima`` slopes are SOLVAX's, which agree with SciPy's ``PchipInterpolator`` and
``Akima1DInterpolator`` to roundoff. ``cubic`` is the mean-of-secants rule NTX
used before, kept so a regression against the old behaviour stays possible; it
matched the previous dependency to 4e-15 when the switch was made.
``tests/test_interp.py`` pins all three.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from solvax import cubic_spline, cubic_spline_evaluate

__all__ = [
    "Interpolator1D",
    "interp1d",
    "interp2d_at",
    "slopes",
]


# --------------------------------------------------------------- slope rules ---
def _secants(x: Array, f: Array) -> Array:
    """Divided differences between neighbouring nodes.

    Reshapes the spacing to broadcast over trailing axes, so the same routine
    serves scalar and vector-valued data.
    """
    h = jnp.diff(x)
    shape = (-1,) + (1,) * (f.ndim - 1)
    return jnp.diff(f, axis=0) / h.reshape(shape)


def _slopes_cubic(x: Array, f: Array) -> Array:
    """Mean of adjacent secants: the rule NTX used before this module."""
    d = _secants(x, f)
    return jnp.concatenate([d[:1], 0.5 * (d[:-1] + d[1:]), d[-1:]], axis=0)


def _slopes_parabolic(x: Array, f: Array) -> Array:
    """Three-point slope weighted by interval length; exact on quadratics."""
    h = jnp.diff(x)
    d = _secants(x, f)
    shape = (-1,) + (1,) * (f.ndim - 1)
    hl, hr = h[:-1].reshape(shape), h[1:].reshape(shape)
    interior = (hr * d[:-1] + hl * d[1:]) / (hl + hr)
    return jnp.concatenate([d[:1], interior, d[-1:]], axis=0)


_RULES = {
    "cubic": _slopes_cubic,
    "parabolic": _slopes_parabolic,
    "akima": lambda x, f: cubic_spline(x, f, slopes="akima").slopes,
    "pchip": lambda x, f: cubic_spline(x, f, monotone=True).slopes,
}


def slopes(x: Array, f: Array, method: str = "akima") -> Array:
    """Node slopes for ``method``. Exposed so callers can build tables once."""
    try:
        rule = _RULES[method]
    except KeyError:
        raise ValueError(
            f"unknown method {method!r}; expected one of {sorted(_RULES)}"
        ) from None
    if x.shape[0] < 3:
        # Two nodes admit only the secant; one admits nothing but a constant.
        if x.shape[0] == 2:
            d = _secants(x, f)
            return jnp.concatenate([d, d], axis=0)
        return jnp.zeros_like(f)
    return rule(x, f)


# ---------------------------------------------------------------- evaluation ---
def _evaluate(x: Array, f: Array, m: Array, xq: Array) -> Array:
    """Cubic Hermite through ``f`` with node slopes ``m``, at ``xq``.

    All slope rules differ only in how they choose ``m``; SOLVAX evaluates the
    Hermite cubic and continues the end cubics beyond the nodes.
    """
    value = cubic_spline_evaluate(cubic_spline(x, f, slopes=m), xq)
    assert not isinstance(value, tuple)  # one derivative order asked for
    return value


def interp1d(xq: Array, x: Array, f: Array, method: str = "akima") -> Array:
    """Interpolate ``f`` sampled at ``x`` onto ``xq``, extrapolating at the ends.

    ``x`` must be increasing. Extrapolation continues the end cubic, which is
    what the callers here rely on: an equilibrium's half mesh does not reach
    either the axis or the boundary, and the profiles are evaluated at both.
    """
    x, f, xq = jnp.asarray(x), jnp.asarray(f), jnp.asarray(xq)
    return _evaluate(x, f, slopes(x, f, method), xq)


class Interpolator1D:
    """A profile with its slope table built once and evaluated many times.

    Radial profiles are built outside any ``vmap`` and then evaluated on a whole
    radial grid, so the table is amortized and there is nothing to gain from
    working locally. The channel lookups are the opposite case; see
    :func:`interp2d_at`.
    """

    def __init__(self, x: Array, f: Array, method: str = "akima") -> None:
        """Precompute the slopes for the chosen rule.

    The slopes are the expensive part and depend only on the data, so building
    the interpolator once and calling it many times avoids recomputing them per
    query.
        """
        self.x = jnp.asarray(x)
        self.f = jnp.asarray(f)
        self.method = method
        self.m = slopes(self.x, self.f, method)

    def __call__(self, xq: Array) -> Array:
        """Interpolate at the query points."""
        return _evaluate(self.x, self.f, self.m, jnp.asarray(xq))

    def _tree_flatten(self):
        """Split into JAX children (arrays) and static data (the method name).

    The method is static because it selects a code path; tracing it as data
    would make every rule compile to the same branchless program.
        """
        return (self.x, self.f, self.m), (self.method,)

    @classmethod
    def _tree_unflatten(cls, aux, children):
        """Rebuild from flattened parts without re-running __init__.

    Bypasses __init__ via __new__ so the precomputed slopes survive a JAX
    transform rather than being recomputed on every unflatten.
        """
        obj = cls.__new__(cls)
        obj.x, obj.f, obj.m = children
        (obj.method,) = aux
        return obj


jax.tree_util.register_pytree_node(
    Interpolator1D, lambda o: o._tree_flatten(), Interpolator1D._tree_unflatten
)


# ------------------------------------------------------------------ 2D lookup ---
def interp2d_at(
    x: Array,
    y: Array,
    values: Array,
    xq: Array,
    yq: Array,
    method: str = "pchip",
) -> Array:
    """Tensor-product bicubic at a single ``(xq, yq)``, using only its stencil.

    The channel lookups evaluate one point per radius. Building coefficient
    tables over the whole ``(nu, E_r)`` grid to read one value out of them is
    ``O(n_nu * n_er)`` of work discarded; the answer depends only on a four-by-
    four neighbourhood. On production database sizes this is four times faster
    forward and eight times faster through the gradient, at identical values.

    The window is *slid* to stay in range rather than having its indices
    clamped. Clamping repeats a node at the boundary, which gives an interval of
    zero width and an evaluation of ``NaN``.
    """
    x, y, values = jnp.asarray(x), jnp.asarray(y), jnp.asarray(values)
    nx, ny = x.shape[0], y.shape[0]
    if nx < 4 or ny < 4:
        # Too small for a local window; fall back to whole-grid tables.
        return _interp2d_full(x, y, values, xq, yq, method)

    ii = jnp.clip(jnp.searchsorted(x, xq, side="right") - 2, 0, nx - 4) + jnp.arange(4)
    jj = jnp.clip(jnp.searchsorted(y, yq, side="right") - 2, 0, ny - 4) + jnp.arange(4)
    return _bicubic(x[ii], y[jj], values[ii][:, jj], xq, yq, method)


def _interp2d_full(x, y, values, xq, yq, method):
    """Bicubic interpolation at a query point from the whole grid."""
    return _bicubic(x, y, values, xq, yq, method)


def _bicubic(x, y, f, xq, yq, method):
    """Tensor-product bicubic Hermite patch from values, both slopes, and the cross term.

    The cross derivative is built by applying the 1-D slope rule along one axis
    and then the other, so a monotone rule stays monotone in each direction
    separately. The patch is linear in its corner data, so it is evaluated as a
    Hermite cubic in ``x`` of the values and y-slopes, then one in ``y``.
    """
    fx = slopes(x, f, method)
    fy = jnp.swapaxes(slopes(y, jnp.swapaxes(f, 0, 1), method), 0, 1)
    fxy = jnp.swapaxes(slopes(y, jnp.swapaxes(fx, 0, 1), method), 0, 1)
    g = _evaluate(x, jnp.stack([f, fy], -1), jnp.stack([fx, fxy], -1), xq)
    return _evaluate(y, g[:, 0], g[:, 1], yq)
