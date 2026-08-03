"""Runtime configuration helpers."""

from __future__ import annotations

from pathlib import Path

from jax import config as jax_config


def enable_x64(enabled: bool = True) -> None:
    """Enable or disable JAX 64-bit arithmetic.

    NTX enables this at import, because the solver and :class:`~ntx.GridSpec`
    both default to float64 and JAX applies the setting when an array is
    *created*, not when it is used. Deferring the decision to the first solve
    silently truncated any geometry built before it: the documented Python
    entry path produced coefficients accurate to about seven digits instead of
    fifteen, with no error and no visible dtype change in the result.

    Call this with ``False`` before constructing geometry to opt out.
    """

    jax_config.update("jax_enable_x64", enabled)


# NTX is a float64 code: GridSpec defaults to float64 and the solver enables x64
# regardless. JAX fixes an array's precision when it is *created*, and geometry
# is created before the first solve, so deferring the decision truncated the
# documented entry path to single precision. Enabling it when this module is
# imported -- which happens before any NTX array exists -- is what makes that
# path correct. Call enable_x64(False) before constructing geometry to opt out.
enable_x64(True)


def geometry_precision_matches(surface, grid) -> bool:
    """Whether ``surface`` was built at the precision ``grid`` asks for.

    A float32 surface fed to an x64 solve is promoted silently, so the run
    completes, reports float64 dtypes, and is wrong in the eighth digit. This
    is the ordering check that makes that impossible to do by accident.
    """

    import jax.numpy as jnp

    requested = jnp.dtype(grid.dtype)
    for field in ("b_cos", "b", "b_sup_theta", "iota"):
        value = getattr(surface, field, None)
        if value is None or not hasattr(value, "dtype"):
            continue
        if jnp.issubdtype(value.dtype, jnp.floating):
            return bool(jnp.dtype(value.dtype).itemsize >= requested.itemsize)
    return True


def configure_compilation_cache(
    cache_dir: str | Path,
    *,
    minimum_compile_seconds: float = 1.0,
    explain_cache_misses: bool = False,
) -> Path:
    """Configure JAX's optional persistent compilation cache.

    The cache is an optimization only; NTX correctness and acceptable repeated
    runtime do not depend on it. Call this before compiling solver workloads.
    """

    if minimum_compile_seconds < 0.0:
        msg = "minimum_compile_seconds must be non-negative"
        raise ValueError(msg)
    path = Path(cache_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    jax_config.update("jax_compilation_cache_dir", str(path))
    jax_config.update(
        "jax_persistent_cache_min_compile_time_secs",
        float(minimum_compile_seconds),
    )
    jax_config.update("jax_explain_cache_misses", explain_cache_misses)
    return path
