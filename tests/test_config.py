from __future__ import annotations

import jax
import pytest

from ntx import configure_compilation_cache


def test_configure_compilation_cache_creates_resolved_directory(tmp_path):
    cache_dir = tmp_path / "jax-cache"
    old_dir = jax.config.jax_compilation_cache_dir
    old_threshold = jax.config.jax_persistent_cache_min_compile_time_secs
    old_explain = jax.config.jax_explain_cache_misses

    try:
        configured = configure_compilation_cache(
            cache_dir,
            minimum_compile_seconds=0.25,
            explain_cache_misses=True,
        )

        assert configured == cache_dir.resolve()
        assert configured.is_dir()
        assert jax.config.jax_compilation_cache_dir == str(configured)
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.25
        assert jax.config.jax_explain_cache_misses is True
    finally:
        jax.config.update("jax_compilation_cache_dir", old_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", old_threshold)
        jax.config.update("jax_explain_cache_misses", old_explain)


def test_configure_compilation_cache_rejects_negative_threshold(tmp_path):
    with pytest.raises(ValueError, match="non-negative"):
        configure_compilation_cache(tmp_path, minimum_compile_seconds=-1.0)
