from __future__ import annotations

import jax


def pytest_configure(config):
    jax.config.update("jax_enable_x64", True)
