//! # `statscore-distributions`
//!
//! Probability distributions for the `statscore` workspace. Every distribution
//! implements [`ContinuousDistribution`] or [`DiscreteDistribution`] from
//! [`statscore-common`].
//!
//! ## Planned modules
//! - `continuous` — Normal, Student-t, χ², F, Beta, Gamma, Exponential, …
//! - `discrete` — Binomial, Poisson, Negative binomial, Geometric, …
//! - `multivariate` — Multivariate normal, Dirichlet, Wishart, Multivariate t
//!
//! ## Dependencies
//! - [`statscore-common`] — distribution traits
//! - [`statscore-special`] — gamma, beta, erf, bessel
//! - [`statscore-linalg`] — Cholesky for multivariate normal
//! - [`statscore-probability`] — moment identities
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for planned modules and status.
//!
//! ## Status
//! Scaffold crate — implementation pending (Phase 1 MVP).
//!
//! [`ContinuousDistribution`]: statscore_common::ContinuousDistribution
//! [`DiscreteDistribution`]: statscore_common::DiscreteDistribution
//! [`statscore-common`]: https://docs.rs/statscore-common

#![warn(missing_docs)]
#![forbid(unsafe_code)]
