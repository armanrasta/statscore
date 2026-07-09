# statscore-common

Shared foundation for the entire `statscore` workspace: traits, errors, type aliases, and numerically stable helpers.

## Overview

Every downstream crate depends on this crate for a single error type, distribution traits, and common numeric utilities. It intentionally does **not** contain special functions or distribution implementations.

## Modules

| Module | Contents |
|--------|----------|
| `error` | [`StatsError`](../src/error.rs), validation helpers (`require_positive`, `require_finite`, …) |
| `traits` | `ContinuousDistribution`, `DiscreteDistribution`, `HypothesisTest`, `TestResult`, estimators |
| `types` | `Scalar`, `Vector`, `Matrix` (ndarray aliases) |
| `numerics` | `log_sum_exp`, `softmax`, `log1pexp`, `log1mexp`, `logistic` |

## What does NOT live here

- Special functions → `statscore-special`
- Linear algebra → `statscore-linalg`
- Distributions → `statscore-distributions`

## Example

```rust
use statscore_common::{ContinuousDistribution, Result, StatsError, log_sum_exp};

let lse = log_sum_exp(&[1.0_f64.ln(), 2.0_f64.ln(), 3.0_f64.ln()]);
```

## Status

**Complete** (Phase 0). Traits and error types are stable; matrix newtypes may move to `statscore-linalg` later.
