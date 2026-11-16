from __future__ import annotations

import os

import jax
import pytest

from ntx import configure_compilation_cache
from ntx.config import _default_cpu_threads, _enable_default_compilation_cache


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


def test_the_default_cache_respects_an_existing_choice(tmp_path, monkeypatch):
    old_dir = jax.config.jax_compilation_cache_dir
    old_threshold = jax.config.jax_persistent_cache_min_compile_time_secs
    try:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        jax.config.update("jax_compilation_cache_dir", None)
        monkeypatch.setenv("NTX_DISABLE_COMPILATION_CACHE", "1")
        assert _enable_default_compilation_cache() is None
        monkeypatch.delenv("NTX_DISABLE_COMPILATION_CACHE")
        monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "theirs"))
        assert _enable_default_compilation_cache() is None
        monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR")
        chosen = _enable_default_compilation_cache()
        if chosen is not None:  # skipped on macOS with jaxlib < 0.10
            assert chosen == (tmp_path / "ntx" / "jax").resolve()
            assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
            assert _enable_default_compilation_cache() is None  # an existing choice wins
    finally:
        jax.config.update("jax_compilation_cache_dir", old_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", old_threshold)


def test_blas_threads_default_to_at_most_four_and_respect_the_caller(monkeypatch):
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    _default_cpu_threads()
    assert 1 <= int(os.environ["OPENBLAS_NUM_THREADS"]) <= 4
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    _default_cpu_threads()
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
