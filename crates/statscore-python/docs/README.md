# statscore-python

Python bindings for `statscore` via [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs).
Depends on **NumPy** for array inputs/outputs (scalars still work as plain Python floats).

## Status

**Phase 0/1 scaffold — distributions exposed.** Importable package with continuous and discrete wrappers.

## Install (development)

```bash
cd crates/statscore-python
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop
```

## Usage

```python
import numpy as np
import statscore
from statscore.distributions import Normal, Poisson, Binomial

print(statscore.__version__)

n = Normal(0.0, 1.0)
print("Φ(1.96) =", n.cdf(1.96))          # Python float → float
print("z_0.975 =", n.ppf(0.975))

x = np.linspace(-2, 2, 5)
print("pdf(x) =", n.pdf(x))               # ndarray → ndarray
print("samples =", n.rvs(5))              # always returns ndarray

p = Poisson(3.0)
print("P(X≤2) =", p.cdf(2))
print("pmf(k) =", p.pmf(np.arange(5)))

b = Binomial(10, 0.5)
print("mean =", b.mean())
```

## Scalars and arrays

| Input | Output |
|-------|--------|
| `float` / `int` | Python `float` / `int` |
| 1-D `numpy.ndarray` (or list / array-like) | `numpy.ndarray` |

Continuous: `pdf`, `logpdf`, `cdf`, `sf`, `ppf` — float or array.  
Discrete: `pmf`, `logpmf`, `cdf` — int or int/float array; `ppf` — float or float array.  
`rvs(size)` always returns a 1-D NumPy array (`float64` continuous, `int64` discrete).

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

Helpers: `standard_normal()`.

## Demo

```bash
python examples/demo_distributions.py
```
