# statscore-python

Python bindings for `statscore` via [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs).
Depends on **NumPy** for array inputs/outputs (scalars still work as plain Python floats).

## Status

**Phase 0/1 scaffold — distributions + fuzzy exposed.** Importable package with continuous/discrete wrappers and fuzzy sets/logic/stats (scalars + NumPy where applicable).

Performance vs SciPy/NumPy, absolute timings, and how to reproduce benches: **[performance.md](performance.md)**. Use a **release** build for any timing (`maturin develop --release`).

## Install (development)

```bash
cd crates/statscore-python
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release   # required for real speed; debug is much slower
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

## Exposed types (`statscore.fuzzy`)

| Symbol | Role |
|--------|------|
| `TriangularFuzzyNumber(a, m, b)` | Triangle with peak at `m` |
| `TrapezoidalFuzzyNumber(a, m1, m2, b)` | Flat top `[m1, m2]` |
| `fuzzy_and_min` / `fuzzy_and_product` / `fuzzy_and_lukasiewicz` | Fuzzy AND |
| `fuzzy_or_max` / `fuzzy_or_sum` | Fuzzy OR |
| `fuzzy_not` / `implication` | Complement / Mamdani |
| `fuzzy_mean` / `fuzzy_variance` / `fuzzy_correlation` | Fuzzy statistics |

Methods on fuzzy numbers: `membership` (float or ndarray), `core`, `support`, `alpha_cut`, `defuzzify_cog`, `defuzzify_mom`, `defuzzify_weighted`.

```python
from statscore.fuzzy import TriangularFuzzyNumber, fuzzy_mean

warm = TriangularFuzzyNumber(18.0, 22.0, 26.0)
print(warm.membership(20.0))           # 0.5
print(warm.membership(np.linspace(18, 26, 5)))
print(fuzzy_mean([warm, TriangularFuzzyNumber(20.0, 22.0, 24.0)]))
```

## Demo

```bash
python examples/demo_distributions.py
python examples/demo_fuzzy.py
```

## Benchmarks

```bash
pip install scipy scikit-fuzzy
python benches/bench_statscore_numpy.py   # absolute (scalar + ndarray)
python benches/bench_vs_scipy.py
python benches/bench_vs_numpy.py
python benches/bench_vs_skfuzzy.py        # fuzzy vs scikit-fuzzy
```

See [performance.md](performance.md) for recorded release numbers and interpretation.
