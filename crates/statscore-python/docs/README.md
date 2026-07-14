# statscore-python

Python bindings for `statscore` via [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs).

## Status

**Phase 0/1 scaffold — distributions exposed.** Importable package with continuous and discrete wrappers.

## Install (development)

```bash
cd crates/statscore-python
pip install maturin
maturin develop
```

## Usage

```python
import statscore
from statscore.distributions import Normal, Poisson, Binomial

print(statscore.__version__)

n = Normal(0.0, 1.0)
print("Φ(1.96) =", n.cdf(1.96))
print("z_0.975 =", n.ppf(0.975))
print("samples =", n.rvs(5))

p = Poisson(3.0)
print("P(X≤2) =", p.cdf(2))
print("pmf(3) =", p.pmf(3))

b = Binomial(10, 0.5)
print("mean =", b.mean())
```

## Exposed types (`statscore.distributions`)

| Class | Constructor |
|-------|-------------|
| `Normal` | `(loc, scale)` |
| `Uniform` | `(a, b)` |
| `Exponential` | `(rate)` |
| `Gamma` | `(shape, scale)` |
| `Beta` | `(alpha, beta)` |
| `ChiSquared` | `(df)` |
| `StudentT` | `(df)` |
| `F` | `(dfn, dfd)` |
| `Binomial` | `(n, p)` |
| `Poisson` | `(lambda)` |
| `Geometric` | `(p)` |

Continuous methods: `pdf`, `logpdf`, `cdf`, `sf`, `ppf`, `mean`, `var`, `std`, `rvs(size=1)`.

Discrete methods: `pmf`, `logpmf`, `cdf`, `ppf`, `mean`, `var`, `rvs(size=1)`.

Helpers: `standard_normal()`.

## Demo

```bash
python examples/demo_distributions.py
```
