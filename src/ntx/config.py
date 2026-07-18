"""Runtime configuration helpers."""

from __future__ import annotations

from pathlib import Path

from jax import config as jax_config


def enable_x64(enabled: bool = True) -> None:
    """Enable or disable JAX 64-bit arithmetic."""

    jax_config.update("jax_enable_x64", enabled)


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
