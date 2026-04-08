# GPU Runs

NTX uses the same JAX solver path on CPU and GPU. The repository keeps GPU
checks separate from the default CPU CI because standard hosted CI runners do
not provide JAX GPU devices.

## GPU Test Targets

- `tests/test_gpu_smoke.py`
- `scripts/run_gpu_regression.py`
- `scripts/sh_office_gpu_smoke.sh`

The GPU smoke tests cover one DKES case and one VMEC case. The regression script
adds device reporting, compile and steady-state timings, and coefficient deltas
against repository reference values.

## Office Workflow

Typical GPU validation sequence:

```bash
sh office
cd /Users/rogeriojorge/local/.NTX
python -m pip install -e ".[dev,docs,io]"
scripts/sh_office_gpu_smoke.sh
```

That wrapper runs:

```bash
python -m pytest -m gpu -q
python scripts/run_gpu_regression.py --output-json gpu-smoke-results.json
```

## Output

`gpu-smoke-results.json` contains:

- JAX backend and visible GPU devices
- per-case compile-plus-first-run timing
- per-case steady-state timing
- solved coefficients
- reference coefficients
- per-coefficient deltas
- max relative error per case

The regression script exits nonzero if:

- no GPU device is visible to JAX
- any coefficient exceeds the configured relative-error tolerance

## Notes

- Keep `x64 = true` for the physics runs unless you are deliberately testing
  reduced precision.
- The GPU references are currently defined for:
  - DKES W7-X smoke case on `tests/fixtures/w7x_eim_sample.ddkes2.data`
  - VMEC W7-X smoke case on `tests/fixtures/wout_w7x_standardConfig.nc`
