//! # `statscore-python`
//!
//! Python bindings for the `statscore` workspace via PyO3. Bindings ship
//! **in parallel with each Rust crate** — not as a final-phase add-on.
//!
//! ## Planned modules
//! - `distributions` — `#[pyclass]` wrappers for each distribution
//! - `hypothesis` — test results as Python dicts
//! - `regression` — `fit()` / `predict()` API
//! - `convert` — zero-copy ndarray ↔ NumPy helpers
//!
//! ## Dependencies
//! - Domain crates (`statscore-distributions`, `statscore-hypothesis`, …)
//! - `pyo3` — Python FFI (this is the **only** crate where `unsafe` is permitted)
//! - `numpy` — NumPy array interop
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for planned modules and status.
//!
//! ## Status
//! Scaffold crate — PyO3/maturin setup pending (Phase 0).
//!
//! ## Example
//! ```ignore
//! import statscore
//! from statscore.distributions import Normal
//! dist = Normal(0.0, 1.0)
//! dist.cdf(1.96)
//! ```

#![warn(missing_docs)]
