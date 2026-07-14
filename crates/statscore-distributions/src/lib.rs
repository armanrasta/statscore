//! # `statscore-distributions`
//!
//! Probability distributions for the `statscore` workspace. Every distribution
//! implements [`ContinuousDistribution`] or [`DiscreteDistribution`] from
//! [`statscore-common`].
//!
//! ## Modules
//! - [`continuous`] — Normal, Uniform, Exponential, Gamma, Beta, χ², Student-t, F
//! - [`discrete`] — Binomial, Poisson, Geometric
//!
//! ## Dependencies
//! - [`statscore-common`] — distribution traits
//! - [`statscore-special`] — gamma, beta, erf for CDFs/PPFs
//! - [`rand`] / [`rand_distr`] — sampling
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for API tables, parameterization,
//! and accuracy notes.
//!
//! ## Example
//! ```
//! use statscore_common::ContinuousDistribution;
//! use statscore_distributions::Normal;
//!
//! let n = Normal::standard();
//! assert!((n.cdf(0.0) - 0.5).abs() < 1e-12);
//! let q = n.ppf(0.975).unwrap();
//! assert!((q - 1.959963984540054).abs() < 1e-10);
//! ```
//!
//! [`ContinuousDistribution`]: statscore_common::ContinuousDistribution
//! [`DiscreteDistribution`]: statscore_common::DiscreteDistribution

#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod continuous;
pub mod discrete;

mod util;

pub use continuous::{
    Beta, ChiSquared, Exponential, FDistribution, Gamma, Normal, StudentT, Uniform,
};
pub use discrete::{Binomial, Geometric, Poisson};
