
# statscore

[//]: # ([![CI Status]&#40;https://github.com/your-org/statscore/workflows/ci.yml/badge.svg&#41;]&#40;https://github.com/your-org/statscore/actions&#41;)

[//]: # ([![Crates.io Version]&#40;https://img.shields.io/crates/v/statscore.svg&#41;]&#40;https://crates.io/crates/statscore&#41;)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Rust Version](https://img.shields.io/badge/rustc-1.82+-blue.svg)](https://blog.rust-lang.org/2024/10/17/Rust-1.82.0.html)

A high-performance, pure-Rust statistical computing engine designed for web, cloud, data science, and edge environments. 

`statscore` is structured as a strictly decoupled Directed Acyclic Graph (DAG) of highly specialized workspaces. It provides robust statistical distributions, linear algebra primitives, hypothesis tests, and modeling frameworks with **zero system dependencies** (no vendor BLAS/MKL required by default).

---

## 🏗️ Architecture
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
      │(all dists)       │
      └────────┬─────────┘
               │
    ┌──────────┼──────────┐
               ▼
     [ Specialized Crates ]
 (hypothesis, regression, etc.)
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
┌────────────┐ ┌────────┐ ┌────────────┐
│statscore-  │ │stats-  │ │statscore   │
│python      │ │core-   │ │(meta-crate)│
│(PyO3)      │ │wasm    │ │            │
└────────────┘ └────────┘ └────────────┘


```
### 📦 Workspace Crate Registry

* [`statscore-common`](crates/statscore-common): Shared domain traits (`Distribution`, `Estimator`), error boundaries, and newtype bounds. **Strictly `forbid(unsafe_code)`**.
* [`statscore-special`](crates/statscore-special): Bedrock mathematical primitives ($\Gamma$, $\text{erf}$, $\text{Bessel}$, $\text{Beta}$) derived using pure math via `libm`.
* [`statscore-linalg`](crates/statscore-linalg): Matrix decompositions (Cholesky, QR, SVD) built on `nalgebra` for pure-Rust portability, with optional BLAS hardware acceleration features.
* [`statscore-probability`](crates/statscore-probability): High-order raw and central moments, Characteristic Functions, and analytical inequality limits (Hoeffding, Chernoff).
* [`statscore-distributions`](crates/statscore-distributions): 30+ continuous, discrete, and multivariate probability distributions with numerically stable `log_pdf` implementations.
* **Specialized Domain Crates**: Multi-crate engine divisions including `descriptive`, `hypothesis`, `regression`, `multivariate`, `timeseries`, `bayesian`, `simulation`, `survival`, `categorical`, `quality`, and `information`.
* [`statscore-python`](crates/statscore-python): Zero-copy PyO3 extension module compiling to high-performance Python bindings, validated direct against SciPy/NumPy.
* [`statscore`](crates/statscore): The top-level meta-crate re-exporting all sub-modules under a unified API.

---

## 🎯 Key Design Philosophies

1. **No System Dependency Hell:** Traditional Rust numeric stacks rely heavily on `ndarray-linalg` or external C/Fortran wrappers like OpenBLAS, causing frequent build breakages on Windows, WASM, and alpine-Docker targets. `statscore` resolves this by utilizing `nalgebra` for pure-Rust matrix routines, making compiling everywhere seamless out-of-the-box.
2. **Explicit Stability Over Convenience:** Loose raw type aliases like `pub type Matrix = Array2<f64>` drop semantic meaning and construction validation. `statscore` leverages custom newtype wrappers to enforce invariants (e.g., positive-definiteness, structural symmetry) directly at compile-time and construction.
3. **No Hidden Panics:** Statistical engineering requires total control over bounds. `statscore` enforces a strict propagation pattern (`?`). Functions do not `unwrap()` or crash on invalid numbers, or near-singular matrices—they return explicit `StatsError` structures.
4. **Isolated PyO3 FFI:** `unsafe_code` is strictly **forbidden** in every single internal numeric crate. Memory-unsafe layout casting and Python FFI boundary handling are completely isolated inside the `statscore-python` layer.

---

## 🚀 Quickstart

Add the meta-crate to your `Cargo.toml`:

```toml
[dependencies]
statscore = "0.1.0"
rand = "0.8"

```

### Example: Working with Distributions

```rust
use statscore::distributions::{Normal, Continuous};
use statscore::common::traits::Distribution;
use rand::thread_rng;

fn main() -> Result<(), statscore::common::error::StatsError> {
    // Construct a normal distribution with strict parameter validation
    let dist = Normal::new(0.0, 1.0)?;
    
    // Evaluate stable mathematical evaluations
    let density = dist.pdf(1.5);
    let log_density = dist.log_pdf(1.5);
    let cumulative = dist.cdf(1.5);
    
    println!("PDF: {}, Log-PDF: {}, CDF: {}", density, log_density, cumulative);

    // Compute precise quantiles safely (Percent-Point Function)
    let median = dist.ppf(0.5)?;
    assert_eq!(median, 0.0);

    // Sample from the distribution
    let mut rng = thread_rng();
    let sample = dist.sample(&mut rng);
    
    Ok(())
}

```

---

## 🛡️ Non-Negotiable Quality Gates

Every implementation iteration must pass the workspace quality standards before structural integration:

* **Correctness Delta:** Values generated from `statscore-special` and `statscore-distributions` are cross-evaluated against R and SciPy ecosystems on every integration pass. High-precision functions must map to `< 1e-12` relative error limits.
* **Property Testing:** Edge combinations, zero boundaries, and mathematical limits are stressed via `proptest` suites using 100,000 continuous random configurations per target module.
* **Zero Allocations in Hot Paths:** Matrix mutations, transformations, and sampling routines execute completely inside allocated workspaces or borrow-sliced slices. Memory usage clean paths are verified using micro-benchmarks.
* **Total Documentation Integration:** Public structural models require mathematical definitions written out in LaTeX syntax accompanied by fully verifiable code examples (`cargo test --doc`).

---

## 🛠️ Performance Hardware Acceleration

For cloud instances or specialized performance pipelines requiring hardware-specific vector acceleration, opt into your target native engine using feature flags on `statscore` or `statscore-linalg`:

```toml
[dependencies]
# Toggle optimized hardware acceleration backends via nalgebra flags
statscore = { version = "0.1.0", features = ["openblas"] }

```

Available acceleration targets: `blas`, `openblas`, `mkl`.

---

## 📜 License

Licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](https://www.google.com/search?q=LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](https://www.google.com/search?q=LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your preference.

