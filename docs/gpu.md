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
cd /path/to/NTX
python -m pip install -e ".[dev,docs,io]"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
scripts/sh_office_gpu_smoke.sh
```

That wrapper runs:

```bash
python -m pytest -m gpu -q
python scripts/run_gpu_regression.py --output-json gpu-smoke-results.json
```

The repository GPU entrypoints also default `XLA_PYTHON_CLIENT_PREALLOCATE=false`
internally. That keeps the shared office GPUs from reserving most device memory
up front and avoids the cuFFT and cubin-load allocation failures seen during
the initial hardware validation.

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
- For runtime profiling, select the backend explicitly:

```bash
python scripts/profile_runtime.py --backend gpu --output-json runtime-profile.json
```

- The GPU references are currently defined for:
  - DKES W7-X smoke case on `tests/fixtures/w7x_eim_sample.ddkes2.data`
  - VMEC W7-X smoke case on `tests/fixtures/wout_w7x_standardConfig.nc`
- Office hardware validation on 2026-04-10 closed with:
  - smoke pytest: `2 passed`
  - DKES smoke max relative error: `9.44e-09`
  - VMEC smoke max relative error: `1.03e-12`
  - steady DKES smoke solve: about `5.53e-02 s`
  - steady VMEC smoke solve: about `5.63e-02 s`
