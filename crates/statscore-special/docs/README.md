# statscore-special

Special mathematical functions: the numerical bedrock of all downstream crates.

## Overview

Provides gamma, beta, error function, modified Bessel, and combinatorics in `f64`. Functions return `NaN`/`±∞` for out-of-domain inputs (SciPy convention) — no `Result` at this layer.

## Modules

| Module | Key functions |
|--------|---------------|
| `gamma` | `ln_gamma`, `gamma`, `digamma`, `trigamma`, `gammainc`, `gammaincc` |
| `beta` | `ln_beta`, `beta`, `betainc`, `betaincinv` |
| `erf` | `erf`, `erfc`, `erf_inv`, `erfc_inv` |
| `bessel` | `i0`, `i1`, `k0`, `k1`, scaled variants, `ln_i0` |
| `combinatorics` | `ln_factorial`, `ln_choose`, `choose`, `ln_perm` |

## Accuracy targets

| Function family | Target |
|-----------------|--------|
| Gamma, beta, erf | ~1e-12–1e-15 vs SciPy |
| Bessel I₀/I₁ | ~1e-13 |
| Bessel K₀/K₁ | ~1e-7 (A&S; upgrade planned) |

## Example

```rust
use statscore_special::{gamma, erf, ln_gamma};

assert!((gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-12);
assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-14);
```

## Status

**Complete** (Phase 0). SciPy validation harness integration pending.
