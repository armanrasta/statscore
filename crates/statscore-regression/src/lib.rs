//! # `statscore-regression`
//!
//! Regression models: ordinary and generalized least squares, GLMs, and
//! regularized regression with diagnostics.
//!
//! ## Planned modules
//! - `linear` — OLS (via QR), WLS, GLS
//! - `glm` — logit/probit links, Binomial/Poisson/Gaussian families, IRLS
//! - `regularized` — Ridge, Lasso, Elastic Net
//! - `diagnostics` — residuals, leverage, Cook's D, VIF
//!
//! ## Dependencies
//! - [`statscore-common`] — [`ModelEstimator`] trait, errors
//! - [`statscore-linalg`] — QR, Cholesky, SVD for OLS/regularization
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for planned modules and status.
//!
//! ## Status
//! Scaffold crate — implementation pending (Phase 2).
//!
//! [`ModelEstimator`]: statscore_common::ModelEstimator

#![warn(missing_docs)]
#![forbid(unsafe_code)]
