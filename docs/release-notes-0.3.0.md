# NTX 0.3.0 Release Notes

NTX `0.3.0` fixes an accuracy defect in the documented Python entry path, adds a
certified adjoint window, and repairs two gates that were passing without
checking anything. It is a minor version rather than a patch because the
precision default is a behaviour change and one previously silent situation now
raises.

## Upgrade first if you use the Python API

Through `0.2.4`, the documented quick start returned coefficients accurate to
about seven digits while reporting `float64`.

`GridSpec` defaults to `float64` and the solver enables JAX's x64 mode, but JAX
fixes an array's precision when it is **created**, and geometry is created
first. So `example_surface()` — and any surface loaded before the first solve —
was silently truncated to single precision and then promoted by the solve. The
run completed, the result claimed `float64`, and it was wrong in the eighth
digit:

| coefficient | `0.2.4` | correct | relative error |
| --- | --- | --- | --- |
| `D11` | 0.0037268260089 | 0.00372682622341 | 5.8e-08 |
| `D31` | -0.222360299995 | -0.222360301809 | 8.2e-09 |
| `D33` | 420.865970528 | 420.865967629 | 6.9e-09 |

The CLI was unaffected: it enables x64 before loading geometry.

x64 is now enabled when `ntx.config` is imported, which happens before any NTX
array exists, so the ordering cannot go wrong. A surface built at a narrower
precision than its grid requests is now **refused** rather than promoted, which
catches every ordering rather than only the one fixed here.

Deliberate single precision still works. Call `ntx.enable_x64(False)` before
constructing geometry and pass a matching grid:

```python
ntx.enable_x64(False)
surface = ntx.example_surface()
result = ntx.solve_monoenergetic(
    surface, ntx.GridSpec(9, 9, 8, dtype="float32", x64=False), case
)
```

## New: a window you can prove

`certify_adjoint_window` returns the smallest adjoint window whose relative
gradient error is *provably* within a requested tolerance, using the certificate
added in SOLVAX `0.12.0`.

```python
window = ntx.certify_adjoint_window(prepared, case, rtol=1e-6)
int(window)                        # 25
window.certified_relative_error    # 3.2e-07, the proven bound
window.status                      # "certified", or "full-window" if none exists
```

The cotangent the certificate needs is exact rather than estimated, because the
transport coefficients are linear functionals of the three retained Legendre
modes — so it costs no extra solve.

Its limits are part of the contract. The bound is a worst case at every step, so
the certified window is wider than the shortest that would have worked; and
where the chain does not localize it returns the *exact* window, which is
correct and saves nothing. On a weakly collisional surface
`advise_adjoint_window` will often suggest something much shorter that happens
to work — it simply cannot prove it. The two are complementary, and neither
replaces widening the window until the gradient stops moving.

Requires `solvax >= 0.12.0`, which the package now pins.

## Two gates that were not checking anything

**The type gate.** `python_version = "3.10"` made mypy parse the installed numpy
stubs under 3.10 rules, where they use 3.12 type syntax. It failed inside numpy
and stopped before reaching a single NTX file. With the pin dropped it checks
all 112 source files, and immediately found a real `tomllib` redefinition, now
fixed.

**DKX discovery.** `find_sfincs_jax_root` looked for a directory named
`sfincs_jax`, which has not existed since the rename to DKX. A present checkout
was therefore not found, and the comparisons that need it skipped silently —
which reads exactly like a pass in a summary. `find_dkx_root` honours `DKX_ROOT`
and both directory names; the old function remains as a deprecated alias.

## The test suite is deterministic again

Two tests regenerated committed evidence in place under `docs/_static`, so a run
left the working tree dirty and the next run started from a tree the previous one
had modified. Identical checkouts disagreed with each other.

The generators now take an output location — `build_manuscript_artifacts.py
--output-dir` and `build_closure_validation_report.py --output-prefix`, both
defaulting to the committed path so deliberate regeneration is unchanged — and
the tests pass a temporary one. A guard test compares tracked-file status
against a collection-time snapshot, so it reports what the *run* dirtied without
failing on a developer's uncommitted edits.

Two consecutive full runs of an unmodified tree now give the same result and
leave the tree clean.

## Documentation

The README leads with a measurement rather than a claim: a design gradient costs
one adjoint solve regardless of parameter count — flat at 89 ms from one
parameter to thirty-two — while central differences grow from 42 ms to 1269 ms,
and the adjoint is five orders of magnitude more accurate. Reproduce with
`python benchmarks/bench_design_derivatives.py --params 1,2,4,8,16,32`.

That comparison is NTX against finite differences on NTX's own solve. It is a
statement about the cost model, not a benchmark against other neoclassical
codes, none of which were run.

The docs build is also clean under `-W` again, which it had not been.

## Release checks

- 518 tests pass, twice in a row on an unmodified tree, leaving it clean
- `ruff check .` clean
- `mypy src/ntx` clean across 112 source files
- `python -m sphinx -W -b html docs docs/_build/html` clean
- fresh clone, clean install, CLI and both README snippets run and reproduce the
  documented values
