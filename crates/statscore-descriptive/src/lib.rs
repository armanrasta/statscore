//! # `statscore-descriptive`
//!
//! Descriptive statistics: univariate summaries, bivariate measures, rank
//! correlations, and robust estimators.
//!
//! ## Planned modules
//! - `univariate` — mean, variance, std, skewness, kurtosis, quantiles
//! - `bivariate` — covariance, Pearson correlation
//! - `rank` — Spearman ρ, Kendall τ
//! - `robust` — median, IQR, MAD, trimmed mean
//! - `grouped` — by-group statistics (parallel via rayon)
//!
//! ## Dependencies
//! - [`statscore-common`] — types, errors, validation helpers
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for planned modules and status.
//!
//! ## Status
//! Scaffold crate — implementation pending (Phase 1 MVP).
//!
//! [`statscore-common`]: https://docs.rs/statscore-common

#![warn(missing_docs)]
#![forbid(unsafe_code)]
