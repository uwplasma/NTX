"""Choosing the adjoint window: a cheap estimate, and a certified one.

Both answer "how many Legendre rows must the reverse pass keep", and they
answer it differently on purpose. :func:`advise_adjoint_window` reads the
operator and costs a norm estimate per row; :func:`certify_adjoint_window`
returns a window with a proof, and costs a differentiated solve. Neither
dominates: on a weakly collisional chain the cheap estimate often suggests a
much shorter window than the certificate can justify, and the certificate is
the one that tells you when it cannot justify anything.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import solvax
from jax import Array

from ._solver_context import _operator_context
from ._solver_factorization import (
    _conditioned_operator_blocks,
    _parameterized_block_fn,
    _solve_modes_with_tail_residual,
)
from ._solver_types import MonoenergeticCase, PreparedMonoenergeticSystem
from .operators import OperatorContext, block_parameters, source_modes
from .transport import coefficients_from_modes

KEEP_LOWEST = 3  # the transport coefficients read Legendre modes 0, 1 and 2
COEFFICIENTS = ("D11", "D31", "D13", "D33")


def advise_adjoint_window(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
):
    """Estimate where the Legendre chain becomes localized enough to truncate.

    Returns SOLVAX's ``LocalizationWindow``: the per-row transfer norms
    ``rho_k``, the first row where they fall below one, and a suggested window.

    The estimate is read from the operator, not from a gradient, and it is not
    a certificate: ``certified`` is always ``False``. The physics behind it is
    that pitch-angle scattering damps Legendre mode ``l`` like ``nu*l(l+1)``
    while the streaming coupling grows only like ``l``, so the chain contracts
    faster the higher one climbs, and the row where it starts contracting moves
    outward as collisions weaken. Use the value as a starting point and widen
    it until the gradient stops moving.
    """

    return solvax.localization_crossover_window(
        lambda k: _conditioned_operator_blocks(ctx, k, d_theta, d_zeta),
        n_xi + 1,
        keep_lowest=3,
    )


def certify_adjoint_window(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    *,
    rtol: float = 1.0e-6,
    coefficient: str = "D11",
):
    """Smallest adjoint window whose gradient error is *provably* within ``rtol``.

    :func:`ntx.advise_adjoint_window` reads the operator and returns a
    plausible window; this returns one with a proof attached. The difference
    matters when the gradient drives an optimizer, because a window chosen by
    eye is wrong by an amount nobody measured.

    A certificate is a statement about one differentiated quantity, so it needs
    that quantity's cotangent rather than the operator alone. The transport
    coefficients are linear functionals of the three retained Legendre modes,
    so the cotangent follows exactly from differentiating
    :func:`ntx.transport.coefficients_from_modes` -- no extra solve, and no
    approximation of the thing being certified.

    Args:
        prepared: geometry from :func:`ntx.prepare_monoenergetic_system`.
        case: the collisionality and electric field to certify at. Both matter:
            weaker collisionality pushes the crossover outward, so the window
            certified at one collisionality is not certified at another.
        rtol: target relative error of the parameter gradient.
        coefficient: which coefficient's gradient to certify --- ``"D11"``,
            ``"D31"``, ``"D13"`` or ``"D33"``.

    Returns:
        SOLVAX's ``LocalizationWindow`` with ``certified=True``. It converts to
        ``int``, so it passes straight to ``adjoint_window=``.
        ``certified_relative_error`` is the proven bound and ``status`` says
        whether a proper window was found or the exact one was returned.

    Note:
        Every step of the bound is a worst case, so the certified window is
        wider than the shortest that would have worked --- by a couple of rows
        on a well-localized chain, by considerably more when it barely
        contracts. Where the chain does not localize the exact window comes
        back, which is correct and saves nothing.
    """

    if coefficient not in COEFFICIENTS:
        msg = f"coefficient must be one of {COEFFICIENTS}; got {coefficient!r}"
        raise ValueError(msg)
    index = COEFFICIENTS.index(coefficient)

    grid = prepared.grid
    geom = prepared.geometry
    epsi_hat = case.resolved_epsi_hat(geom.transport_psi_scale)
    ctx = _operator_context(prepared.surface, geom, grid, case.nu_hat, epsi_hat)
    s1, s3 = source_modes(ctx, grid.n_xi)
    rhs_low = jnp.stack((s1[:KEEP_LOWEST], s3[:KEEP_LOWEST]), axis=-1)

    f1_modes, f3_modes, _ = _solve_modes_with_tail_residual(
        ctx, grid.n_xi, prepared.d_theta, prepared.d_zeta, s1, s3, None
    )
    retained = jnp.stack((f1_modes, f3_modes), axis=-1)

    def selected(modes: Array) -> Array:
        values = coefficients_from_modes(
            geom, modes[..., 0], modes[..., 1], ctx.nu_hat
        )
        return jnp.asarray(values[index])

    cotangent = jax.grad(selected)(retained)

    return solvax.certified_adjoint_window(
        _parameterized_block_fn(prepared.d_theta, prepared.d_zeta),
        grid.n_xi + 1,
        KEEP_LOWEST,
        block_parameters(ctx),
        rhs_low,
        cotangent,
        rtol=rtol,
    )
