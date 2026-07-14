//! # `statscore-fuzzy`
//!
//! Fuzzy statistics for the `statscore` workspace
//!
//! Where probability models *randomness* (`P(X = 5)`), fuzzy sets model
//! *imprecision* (`μ("approximately 5") ∈ [0, 1]`). This crate provides the
//! Phase-1 core: fuzzy sets/numbers, fuzzy logic operators, and basic fuzzy
//! statistics, all built on [`statscore-common`] only.
//!
//! ## Modules
//! - [`traits`] — [`FuzzySet`](traits::FuzzySet) and
//!   [`FuzzyNumber`](traits::FuzzyNumber)
//! - [`sets`] — [`TriangularFuzzyNumber`](sets::TriangularFuzzyNumber),
//!   [`TrapezoidalFuzzyNumber`](sets::TrapezoidalFuzzyNumber)
//! - [`logic`] — [`FuzzyLogic`](logic::FuzzyLogic) t-norms / t-conorms
//! - [`statistics`] — fuzzy mean, variance, correlation
//!
//! ## Example
//! ```
//! use statscore_fuzzy::sets::TriangularFuzzyNumber;
//! use statscore_fuzzy::traits::{FuzzySet, FuzzyNumber};
//!
//! let warm = TriangularFuzzyNumber::new(18.0, 22.0, 26.0).unwrap();
//! assert_eq!(warm.membership(22.0), 1.0);
//! assert_eq!(warm.membership(20.0), 0.5);
//! assert_eq!(warm.membership(30.0), 0.0);
//! assert!((warm.defuzzify_cog() - 22.0).abs() < 1e-12);
//! ```
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for concepts and roadmap.
//!
//! [`statscore-common`]: https://docs.rs/statscore-common

#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod logic;
pub mod sets;
pub mod statistics;
pub mod traits;

pub use logic::FuzzyLogic;
pub use sets::{TrapezoidalFuzzyNumber, TriangularFuzzyNumber};
pub use statistics::{fuzzy_correlation, fuzzy_mean, fuzzy_variance};
pub use traits::{FuzzyNumber, FuzzySet};
