//! # `statscore-timeseries`
//!
//! Time series analysis: stationarity tests, exponential smoothing, ARIMA
//! modelling, and seasonal decomposition.
//!
//! ## Planned modules
//! - `stationarity` — Augmented Dickey–Fuller, KPSS
//! - `smoothing` — EMA, Holt, Holt–Winters
//! - `arima` — AR, MA, ARMA, ARIMA
//! - `decomposition` — trend + seasonal + residual
//!
//! ## Dependencies
//! - [`statscore-common`] — types and errors
//! - [`statscore-linalg`] — linear algebra for ARMA estimation
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for planned modules and status.
//!
//! ## Status
//! Scaffold crate — implementation pending (Phase 2).
//!
//! [`statscore-common`]: https://docs.rs/statscore-common

#![warn(missing_docs)]
#![forbid(unsafe_code)]
