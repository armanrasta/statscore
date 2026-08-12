# statscore

**Pure-Rust statistics library with Python bindings** — probability distributions, linear algebra, time series forecasting (ARIMA, ETS), and fuzzy sets. A fast, dependency-light alternative for SciPy-style work in Rust and Python.

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Rust Version](https://img.shields.io/badge/rustc-1.82+-blue.svg)](https://blog.rust-lang.org/2024/10/17/Rust-1.82.0.html)
[![Benchmarks](https://img.shields.io/badge/benchmarks-nightly%20CI-orange.svg)](benchmarks/README.md)

`statscore` is a modular Rust workspace for **statistical computing** and **quantitative research**: numerically stable `pdf` / `log_pdf` / `cdf` / `ppf`, pure-Rust matrix factorizations, forecasting baselines + ETS + ARIMA, and PyO3/NumPy bindings. Default builds need **no OpenBLAS/MKL**.

| You want… | Start here |
|-----------|------------|
| Rust distributions API | [`crates/statscore-distributions`](crates/statscore-distributions) |
| Python package | [`crates/statscore-python`](crates/statscore-python) |
| Time series / ARIMA / ETS | [`crates/statscore-timeseries`](crates/statscore-timeseries) |
| Speed vs SciPy / statsmodels | [`benchmarks/`](benchmarks/README.md) |
| Roadmap | [Issue #1](https://github.com/armanrasta/statscore/issues/1) |

---

## What is statscore?

`statscore` is an open-source **statistics and data-science toolkit** written in Rust, with optional **Python bindings** (PyO3 + maturin + NumPy). It targets developers who need:

- A **SciPy-like** statistics surface without leaving Rust
- **Fast Python** kernels for pdf/cdf and forecasting vs SciPy / statsmodels
- **Quant-friendly** building blocks (distributions, linalg, timeseries, fuzzy sets)
- Portable builds (Linux, macOS, Windows; WASM-friendly pure-Rust linalg by default)

---

## Features

- **Probability distributions** — Normal, Gamma, Beta, Student-t, χ², F, Exponential, Uniform, Binomial, Poisson, Geometric (expanding catalog)
- **Special functions** — γ, erf, beta, Bessel-class primitives via pure math (`libm`)
- **Linear algebra** — Cholesky, QR, SVD and solvers on `nalgebra` (optional BLAS features)
- **Time series forecasting** — naive / drift, ETS (ANN/AAN/AAA/MNN), ARIMA, Prophet-style OLS, ADF/KPSS, classical decompose, Markov chains
- **Fuzzy sets** — triangular/trapezoidal numbers, fuzzy logic, fuzzy mean/variance/correlation
- **Python package** — `statscore.distributions`, `statscore.fuzzy`, `statscore.timeseries` (scalars + NumPy arrays)
- **Public benchmarks** — median wall times + 95% CI on speedup ratios ([`benchmarks/`](benchmarks/README.md))

---

## How to install (Rust)

Add individual crates (recommended while the meta-crate catches up):

```toml
[dependencies]
statscore-distributions = "0.1.0"
statscore-timeseries = "0.1.0"
statscore-linalg = "0.1.0"
```

Or from git until crates.io publish lands:

```toml
[dependencies]
statscore-distributions = { git = "https://github.com/armanrasta/statscore" }
```

### Rust example: normal distribution

```rust
use statscore_common::ContinuousDistribution;
use statscore_distributions::Normal;

fn main() {
    let n = Normal::standard();
    let density = n.pdf(1.5);
    let log_density = n.log_pdf(1.5);
    let cumulative = n.cdf(1.5);
    let q975 = n.ppf(0.975).unwrap();
    println!("pdf={density} log_pdf={log_density} cdf={cumulative} ppf(0.975)={q975}");
}
```

---

## How to install (Python)

```bash
cd crates/statscore-python
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release   # always release for real speed
```

### Python example: distributions + forecast

```python
import numpy as np
from statscore.distributions import Normal
from statscore.timeseries import drift, EtsFit, ArimaModel

n = Normal(0.0, 1.0)
print(n.cdf(1.96), n.pdf(np.linspace(-2, 2, 5)))

x = np.arange(40.0)
print(drift(x, 5)["point"])
print(EtsFit.fit(x, "AAN").forecast(3)["point"])
print(ArimaModel.fit(x, 1, 1, 0).aic())
```

PyPI wheels are on the roadmap ([#25](https://github.com/armanrasta/statscore/issues/25)).

---

## How does performance compare to SciPy?

Release builds are competitive or faster on many scalar and array distribution ops, and much faster on pragmatic ETS/ARIMA-style workflows vs statsmodels (different estimators — see methodology).

```bash
python benchmarks/scripts/run_suite.py
```

Latest snapshot and nightly CI: [`benchmarks/`](benchmarks/README.md) · details in [`crates/statscore-python/docs/performance.md`](crates/statscore-python/docs/performance.md).

---

## Architecture

```
                ┌─────────────────────────┐
                │    statscore-common      │
                │  traits, errors, types   │
                └──────────┬──────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│statscore-    │  │statscore-    │  │statscore-    │
│linalg        │  │probability   │  │special       │
│(decomp,solve)│  │(moments,ineq)│  │(gamma,erf,   │
└──────┬───────┘  └──────┬───────┘  │ beta,bessel) │
       │                 │          └──────┬───────┘
       └────────┬────────┘                 │
                ▼                          │
      ┌──────────────────┐                 │
      │statscore-        │◄────────────────┘
      │distributions     │
      └────────┬─────────┘
               │
     [ Specialized crates ]
  hypothesis, regression, timeseries,
  fuzzy, bayesian, simulation, …
               │
               ▼
      ┌──────────────────┐
      │statscore-python  │
      │(PyO3 + NumPy)    │
      └──────────────────┘
```

### Workspace crates

* [`statscore-common`](crates/statscore-common) — traits (`Distribution`, …), errors; `forbid(unsafe_code)`
* [`statscore-special`](crates/statscore-special) — γ, erf, beta, Bessel via `libm`
* [`statscore-linalg`](crates/statscore-linalg) — Cholesky, QR, SVD; optional BLAS
* [`statscore-distributions`](crates/statscore-distributions) — continuous + discrete distributions with stable `log_pdf`
* [`statscore-timeseries`](crates/statscore-timeseries) — forecasting + stationarity + decompose
* [`statscore-fuzzy`](crates/statscore-fuzzy) — fuzzy numbers, logic, stats
* [`statscore-python`](crates/statscore-python) — PyO3 bindings
* Scaffolds: descriptive, hypothesis, regression, multivariate, bayesian, simulation, survival, categorical, quality, information, probability

---

## Design principles

1. **No system-dependency hell by default** — pure-Rust `nalgebra` path; optional OpenBLAS/MKL features when you want them.
2. **Explicit errors, no hidden panics** — invalid parameters and singular systems return `StatsError`.
3. **Unsafe isolated** — numeric crates forbid `unsafe`; only `statscore-python` allows it for PyO3.
4. **Docs + examples** — public APIs documented; crate guides under each `crates/*/docs/`.

---

## Optional BLAS acceleration

```toml
[dependencies]
statscore-linalg = { version = "0.1.0", features = ["openblas"] }
```

Available targets depend on crate features (`blas`, `openblas`, `mkl` where enabled).

---

## License

Dual-licensed under:

* [Apache License 2.0](LICENSE-APACHE)
* [MIT license](LICENSE-MIT)

at your option.

---

## Links

* [Roadmap](https://github.com/armanrasta/statscore/issues/1)
* [Benchmarks](benchmarks/README.md)
* [Performance notes](crates/statscore-python/docs/performance.md)
* [Documentation index](docs/README.md)
* [Issues](https://github.com/armanrasta/statscore/issues)
