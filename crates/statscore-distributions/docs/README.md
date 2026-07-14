# statscore-distributions

Probability distributions implementing `ContinuousDistribution` / `DiscreteDistribution` from `statscore-common`.

## Status

**Phase 1 MVP — in progress.** Core univariate distributions implemented; multivariate and remaining continuous/discrete families still planned.

## Implemented

### Continuous

| Distribution | Type | Parameters | Notes |
|--------------|------|------------|-------|
| `Normal` | `Normal` | `loc`, `scale` | via erf / erf_inv |
| `Uniform` | `Uniform` | `a`, `b` | continuous on `[a,b]` |
| `Exponential` | `Exponential` | `rate` λ | mean = 1/λ |
| `Gamma` | `Gamma` | `shape`, `scale` | SciPy/R scale param |
| `Beta` | `Beta` | `alpha`, `beta` | on (0,1) |
| `ChiSquared` | `ChiSquared` | `df` | = Gamma(df/2, 2) |
| `StudentT` | `StudentT` | `df` | location 0, scale 1 |
| `FDistribution` | `FDistribution` | `dfn`, `dfd` | Fisher–Snedecor F |

### Discrete

| Distribution | Type | Parameters | Support |
|--------------|------|------------|---------|
| `Binomial` | `Binomial` | `n`, `p` | `{0,…,n}` |
| `Poisson` | `Poisson` | `lambda` | `{0,1,2,…}` |
| `Geometric` | `Geometric` | `p` | failures before first success `{0,1,…}` |

## API pattern

```rust
use statscore_common::ContinuousDistribution;
use statscore_distributions::Normal;

let n = Normal::new(0.0, 1.0).unwrap();
let density = n.pdf(1.0);
let prob = n.cdf(1.96);
let quantile = n.ppf(0.975).unwrap();
let samples = n.sample(&mut rand::rng(), 1000);
```

Same shape for discrete (`pmf` instead of `pdf`).

## Dependencies

- `statscore-common` — traits, `StatsError`
- `statscore-special` — incomplete gamma/beta, erf
- `rand` / `rand_distr` — sampling

## Planned (not yet)

Logistic, Weibull, Cauchy, Pareto, von Mises, Multinomial, NegativeBinomial, MultivariateNormal, Dirichlet, …

## Python

Bindings live in `statscore-python` (`statscore.distributions`). See that crate’s docs.

## Testing

```bash
cargo test -p statscore-distributions
```

## Benchmarks

```bash
# Fast wall-clock microbench
cargo run -p statscore-distributions --release --example microbench

# Criterion (HTML report under target/criterion/)
cargo bench -p statscore-distributions --bench distributions

# Python vs SciPy (needs venv with scipy + maturin develop)
python crates/statscore-python/benches/bench_vs_scipy.py
```
