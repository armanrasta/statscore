# statscore-fuzzy

Fuzzy sets, fuzzy logic, and fuzzy statistics.

## Overview

Classical probability models **randomness** (`P(X = 5)`). Fuzzy sets model
**imprecision** through a membership function `μ: ℝ → [0, 1]`
(`μ("approximately 5")`). This crate is the standalone, opt-in home for that
machinery. It depends only on `statscore-common`, so it never burdens the core
statistical crates.

## Modules

| Module | Contents |
|--------|----------|
| `traits` | `FuzzySet` (membership, core, support, α-cut) and `FuzzyNumber` (defuzzification) |
| `sets` | `TriangularFuzzyNumber`, `TrapezoidalFuzzyNumber` |
| `logic` | `FuzzyLogic` — t-norms (AND), t-conorms (OR), complement, Mamdani implication |
| `statistics` | `fuzzy_mean`, `fuzzy_variance`, `fuzzy_correlation` |

## Concepts

- **Membership** `μ(x)` — degree of belonging in `[0, 1]`.
- **Core** — `{ x | μ(x) = 1 }`.
- **Support** — closure of `{ x | μ(x) > 0 }`.
- **α-cut** — `{ x | μ(x) ≥ α }`, returned as an interval.
- **Defuzzification** — collapse a fuzzy number to a crisp value:
  - COG (center of gravity / centroid)
  - MOM (mean of maxima)
  - weighted blend of the two

## Dependencies

- `statscore-common` — shared `Result` / `StatsError` and validation helpers.

## Example

```rust
use statscore_fuzzy::sets::TriangularFuzzyNumber;
use statscore_fuzzy::traits::{FuzzySet, FuzzyNumber};
use statscore_fuzzy::statistics::fuzzy_mean;

// "warm" ≈ 22°C
let warm = TriangularFuzzyNumber::new(18.0, 22.0, 26.0)?;
assert_eq!(warm.membership(22.0), 1.0);
assert_eq!(warm.membership(20.0), 0.5);

let data = [
    TriangularFuzzyNumber::new(4.5, 5.0, 5.5)?,
    TriangularFuzzyNumber::new(4.8, 5.1, 5.4)?,
];
let mean = fuzzy_mean(&data)?;
println!("defuzzified mean = {:.3}", mean.defuzzify_cog());
# Ok::<(), statscore_common::StatsError>(())
```

Run the example:

```bash
cargo run -p statscore-fuzzy --example fuzzy_basics
```

## Status

**Phase 1 core — implemented.** Triangular/trapezoidal numbers, fuzzy logic,
and basic fuzzy statistics with unit tests and doctests.

Python: `from statscore.fuzzy import TriangularFuzzyNumber, …` via
`statscore-python` (see that crate’s guide).

### Planned (future phases)

- More membership functions (Gaussian, sigmoidal, singleton).
- Fuzzy inference systems (Mamdani, Sugeno).
- Extensions: `statscore-fuzzy-regression`, `statscore-fuzzy-clustering`.
- Property tests (α-cut monotonicity, defuzzification invariants).
