#!/usr/bin/env python3
"""What the bounded reverse pass costs and saves on the monoenergetic solve.

Differentiates the same solve three ways and reports compiled reverse-mode
temporary memory, wall time, and the gradient itself:

  taped        reverse mode straight through the elimination
  full window  the exact-window rule with every Legendre row retained
  advised      the same rule at the window the localization profile suggests

The full window is exact by construction, so its gradient must match the taped
one to rounding; that is the check that makes it safe as a default. The advised
window trades a quantified error for a reverse pass that stops growing with
the Legendre resolution.

    python benchmarks/bench_exact_window_adjoint.py --n-xi 32,64,128,256
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import ntx  # noqa: E402
from ntx import GridSpec, MonoenergeticCase, advise_adjoint_window, example_surface  # noqa: E402
from ntx._solver import (  # noqa: E402
    _solve_modes_with_tail_residual,
    prepare_monoenergetic_system,
)
from ntx.operators import OperatorContext, source_modes  # noqa: E402


def compiled_temp_bytes(fn, *args) -> int:
    return jax.jit(fn).lower(*args).compile().memory_analysis().temp_size_in_bytes


def median_ms(fn, *args, reps: int = 5) -> float:
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def measure(n_xi: int, n_theta: int, n_zeta: int, nu: float, epsi: float) -> dict:
    grid = GridSpec(n_theta, n_zeta, n_xi)
    prepared = prepare_monoenergetic_system(example_surface(), grid)
    case = MonoenergeticCase(nu, er_hat=epsi)
    resolved = case.resolved_epsi_hat(prepared.geometry.transport_psi_scale)
    ctx = OperatorContext(
        prepared.surface, prepared.geometry, jnp.asarray(nu), jnp.asarray(resolved)
    )
    s1, s3 = source_modes(ctx, grid.n_xi)
    advice = advise_adjoint_window(ctx, grid.n_xi, prepared.d_theta, prepared.d_zeta)

    weights = jax.random.normal(jax.random.PRNGKey(0), (3, prepared.d_theta.shape[0]))

    def loss(nu_scale, window):
        scaled = OperatorContext(
            prepared.surface, prepared.geometry, ctx.nu_hat * nu_scale, ctx.epsi_hat
        )
        f1, _, _ = _solve_modes_with_tail_residual(
            scaled, grid.n_xi, prepared.d_theta, prepared.d_zeta, s1, s3, window
        )
        return jnp.sum(weights * f1[:3])

    one = jnp.asarray(1.0)
    full_rows = grid.n_xi + 1

    def entry(window):
        fn = jax.grad(lambda a: loss(a, window))
        return {
            "compiled_temp_bytes": compiled_temp_bytes(fn, one),
            "ms_median": median_ms(jax.jit(fn), one),
            "gradient": float(fn(one)),
        }

    taped = entry(None)  # None == full window == exact; see below for the tape
    # The taped reference: close the parameters over the generator so SOLVAX
    # cannot use its exact-window rule and JAX records the sweep instead.
    import solvax

    from ntx._solver import _operator_block_fn

    def taped_loss(nu_scale):
        scaled = OperatorContext(
            prepared.surface, prepared.geometry, ctx.nu_hat * nu_scale, ctx.epsi_hat
        )
        rhs = jnp.stack((s1[:3], s3[:3]), axis=-1)
        modes, _ = solvax.block_thomas_truncated_fn_with_residual(
            _operator_block_fn(scaled, prepared.d_theta, prepared.d_zeta),
            n_blocks=full_rows,
            rhs_low=rhs,
            keep_lowest=3,
            residual_rhs_index=0,
        )
        return jnp.sum(weights * modes[..., 0][:3])

    taped_fn = jax.grad(taped_loss)
    taped = {
        "compiled_temp_bytes": compiled_temp_bytes(taped_fn, one),
        "ms_median": median_ms(jax.jit(taped_fn), one),
        "gradient": float(taped_fn(one)),
    }
    full = entry(full_rows)
    advised = entry(advice.window)

    def rel(a, b):
        return abs(a - b) / max(abs(b), 1e-300)

    return {
        "n_xi": n_xi,
        "n_theta": n_theta,
        "n_zeta": n_zeta,
        "block_size_m": int(prepared.d_theta.shape[0]),
        "nu_hat": nu,
        "epsi_hat": epsi,
        "advised_window": int(advice.window),
        "crossover_row": int(advice.crossover_row),
        "localized": bool(advice.localized),
        "taped": taped,
        "exact_full_window": {
            **full,
            "rel_error_vs_taped": rel(full["gradient"], taped["gradient"]),
        },
        "advised_window_result": {
            **advised,
            "rel_error_vs_taped": rel(advised["gradient"], taped["gradient"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-xi", default="32,64,128,256")
    parser.add_argument("--n-theta", type=int, default=9)
    parser.add_argument("--n-zeta", type=int, default=9)
    parser.add_argument("--nu", type=float, default=1.0e-2)
    parser.add_argument("--epsi", type=float, default=1.0e-3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = []
    print("reverse-mode cost of the monoenergetic solve", flush=True)
    for n_xi in [int(v) for v in args.n_xi.split(",")]:
        row = measure(n_xi, args.n_theta, args.n_zeta, args.nu, args.epsi)
        rows.append(row)
        t, f, a = row["taped"], row["exact_full_window"], row["advised_window_result"]
        mib = 2**20
        print(
            f"  N_xi={n_xi:5d} m={row['block_size_m']:4d} advised_w={row['advised_window']:4d}"
            f"   taped {t['compiled_temp_bytes']/mib:7.1f} MiB {t['ms_median']:7.1f} ms"
            f" | full {f['compiled_temp_bytes']/mib:7.1f} MiB {f['ms_median']:7.1f} ms"
            f" (err {f['rel_error_vs_taped']:.1e})"
            f" | advised {a['compiled_temp_bytes']/mib:6.1f} MiB {a['ms_median']:6.1f} ms"
            f" (err {a['rel_error_vs_taped']:.1e})",
            flush=True,
        )

    record = {
        "provenance": {
            "ntx_version": getattr(ntx, "__version__", "unknown"),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "device_kind": jax.devices()[0].device_kind,
            "x64": bool(jax.config.jax_enable_x64),
            "platform": platform.platform(),
        },
        "rows": rows,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
