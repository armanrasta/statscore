//! Estimator traits for point estimation, interval estimation, and model fitting.

use crate::error::Result;

/// An estimator that produces a single value (point estimate) from data.
///
/// # Type parameters
/// - `T` — the data type (usually `f64`)
/// - `E` — the estimate type (usually `f64`, or a struct of parameters)
pub trait PointEstimator<T, E> {
    /// Compute the point estimate from `data`.
    ///
    /// # Errors
    /// [`crate::error::StatsError::InsufficientData`] if data is too small.
    fn estimate(&self, data: &[T]) -> Result<E>;
}

/// An estimator that produces an interval (e.g., confidence interval).
pub trait IntervalEstimator<T, E> {
    /// Compute the interval estimate from `data` at the given `confidence_level`.
    ///
    /// `confidence_level` must be in (0, 1), e.g., 0.95.
    ///
    /// # Errors
    /// - [`crate::error::StatsError::Domain`] if `confidence_level` is not in (0, 1).
    /// - [`crate::error::StatsError::InsufficientData`] if data is too small.
    fn interval(&self, data: &[T], confidence_level: f64) -> Result<(E, E)>;
}

/// A model that can be fit to (input, output) pairs and used for prediction.
///
/// # Type parameters
/// - `X` — input type (feature vector or design matrix)
/// - `Y` — output type (response)
/// - `M` — the fitted model type (e.g., `OlsResult`)
pub trait ModelEstimator<X, Y, M> {
    /// Fit the model to `inputs` and `outputs`.
    ///
    /// # Errors
    /// - [`crate::error::StatsError::DimensionMismatch`] if shapes are incompatible.
    /// - [`crate::error::StatsError::SingularMatrix`] if design matrix is rank-deficient.
    fn fit(&self, inputs: &X, outputs: &Y) -> Result<M>;

    /// Use a fitted model to make predictions on new `inputs`.
    fn predict(&self, model: &M, inputs: &X) -> Result<Y>;
}