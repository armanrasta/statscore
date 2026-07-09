//! # `statscore-survival`
//!
//! Survival analysis: Kaplan–Meier estimation, Cox proportional hazards,
//! log-rank tests, and parametric survival models.
//!
//! ## Planned modules
//! - `kaplan_meier` — product-limit estimator
//! - `cox` — proportional hazards regression
//! - `log_rank` — log-rank test
//! - `parametric` — Weibull and exponential survival models
//!
//! ## Dependencies
//! - [`statscore-common`] — types and errors
//! - [`statscore-distributions`] — Weibull, exponential distributions
//! - [`statscore-regression`] — Cox regression infrastructure
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for planned modules and status.
//!
//! ## Status
//! Scaffold crate — implementation pending (Phase 3).
//!
//! [`statscore-common`]: https://docs.rs/statscore-common

#![warn(missing_docs)]
#![forbid(unsafe_code)]
