# statscore-python

Python bindings for the `statscore` workspace via PyO3. **Bindings ship in parallel with each Rust crate.**

## Overview

This is the only crate where `unsafe` is permitted (PyO3 FFI). Domain logic stays in pure-Rust crates; this crate wraps and exposes them.

## Planned modules

| File | Wraps |
|------|-------|
| `convert.rs` | ndarray ↔ NumPy zero-copy |
| `distributions.rs` | `statscore-distributions` |
| `hypothesis.rs` | `statscore-hypothesis` |
| `regression.rs` | `statscore-regression` |
| … | One file per domain crate |

## Build

```bash
# pyproject.toml + maturin
maturin develop
pip install .
```

## Error mapping

`StatsError` variants → Python exceptions (`ValueError`, `RuntimeError`, …).

## Rollout

| Phase | PyPI version | Modules exposed |
|-------|--------------|-----------------|
| 0 | — | Empty importable package |
| 1 | `0.1.0-alpha` | distributions, descriptive, hypothesis |
| 2 | `0.2.0` | + regression, multivariate, timeseries, … |
| Release | `1.0.0` | Full API |

## Status

**Scaffold** (Phase 0).
