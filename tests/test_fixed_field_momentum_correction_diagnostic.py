from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path("/Users/rogeriojorge/local/NTX/examples/fixed_field_momentum_correction_diagnostic.py")
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "fixed_field_momentum_correction_diagnostic_test",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_assemble_dense_species_matrix_orders_blocks_species_first():
    module = _load_module()
    blocks = np.arange(2 * 2 * 2 * 2, dtype=float).reshape(2, 2, 2, 2)
    dense = module._assemble_dense_species_matrix(blocks)
    expected = np.transpose(blocks, (0, 2, 1, 3)).reshape(4, 4)
    np.testing.assert_allclose(dense, expected)


def test_relative_residual_norm_is_small_for_exact_solution():
    module = _load_module()
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    rhs = np.array([7.0, 5.0])
    solution = np.linalg.solve(matrix, rhs)
    assert module._relative_residual_norm(matrix, rhs, solution) < 1.0e-12
