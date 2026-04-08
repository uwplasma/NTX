"""Runtime configuration helpers."""

from __future__ import annotations

from jax import config as jax_config


def enable_x64(enabled: bool = True) -> None:
    """Enable or disable JAX 64-bit arithmetic."""

    jax_config.update("jax_enable_x64", enabled)
