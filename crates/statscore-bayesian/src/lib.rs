//! # `statscore-bayesian`
//!
//! Bayesian inference: conjugate posterior updates and MCMC samplers with
//! convergence diagnostics.
//!
//! ## Planned modules
//! - `conjugate` — Beta–Binomial, Normal–Normal, Gamma–Poisson, Dirichlet–Multinomial
//! - `mcmc` — Metropolis–Hastings, Gibbs sampling
//! - `diagnostics` — Geweke, Gelman–Rubin
//!
//! ## Dependencies
//! - [`statscore-common`] — types and errors
//! - [`statscore-special`] — gamma, beta for conjugate posteriors
//! - [`statscore-distributions`] — prior and likelihood distributions
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
