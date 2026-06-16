//! # `statscore-common`
//!
//! Shared traits, error types, numeric utilities, and type aliases
//! for the entire `statscore` workspace.
//!
//! ## What lives here
//! - [`error::StatsError`] — the single error type for all crates
//! - [`error::Result`]    — `std::result::Result<T, StatsError>`
//! - [`traits`]           — `ContinuousDistribution`, `DiscreteDistribution`,
//!                          `HypothesisTest`, `TestResult`, `Alternative`, etc.
//! - [`types`]            — `Scalar`, `Vector`, `Matrix`, `VectorView`, `MatrixView`
//! - [`numerics`]         — `log_sum_exp`, `softmax`, `log1pexp`, `log1mexp`,
//!                          `log_add`, `logistic`, `log_logistic`
//!
//! ## What does NOT live here
//! - Special functions (gamma, beta, erf, bessel) → `statscore-special`
//! - Descriptive statistics → `statscore-descriptive`
//! - Any distribution implementations → `statscore-distributions`

#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod error;
pub mod numerics;
pub mod traits;
pub mod types;

// ── Flat re-exports for ergonomic imports ────────────────────────────────────
//
// Downstream crates write:
//   use statscore_common::{Result, StatsError, ContinuousDistribution, ...};
// instead of digging into submodules.

pub use error::{
    Result,
    StatsError,
    // Validation helpers — used constantly in every crate
    require_finite,
    require_in_range,
    require_min_len,
    require_non_negative,
    require_positive,
    require_same_len,
};

pub use types::{Matrix, MatrixView, Scalar, Vector, VectorView};

pub use traits::{
    Alternative,
    ContinuousDistribution,
    DiscreteDistribution,
    FittableDistribution,
    HypothesisTest,
    IntervalEstimator,
    MleFit,
    ModelEstimator,
    MomFit,
    MultivariateContinuousDistribution,
    PointEstimator,
    TestResult,
};

pub use numerics::{
    log_add,
    log1mexp,
    log1pexp,
    log_logistic,
    log_sum_exp,
    logistic,
    softmax,
};