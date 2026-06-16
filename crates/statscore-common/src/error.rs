//! Unified error type for the entire statscore workspace.
//!
//! Every crate maps its internal errors into [`StatsError`] so that
//! callers — including the Python binding layer — only ever deal with
//! one error type.

use thiserror::Error;

/// The single error type used throughout the statscore workspace.
///
/// ## Design rules
/// - All variants carry a human-readable message or structured fields.
/// - No variant is generic over an inner error type; we convert at the boundary.
/// - The Python layer matches on these variants to choose the right Python exception.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum StatsError {
    /// A parameter or input value is outside its mathematical domain.
    ///
    /// Examples: negative standard deviation, probability > 1, k > n in binomial.
    #[error("Domain error: {0}")]
    Domain(String),

    /// Two arrays or matrices have incompatible shapes.
    ///
    /// Carry a human-readable description so callers do not need to re-derive it.
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// An iterative algorithm failed to converge within the allowed iterations.
    #[error("Convergence failed after {iterations} iterations: {message}")]
    Convergence {
        iterations: usize,
        message: String,
    },

    /// A matrix is numerically singular (cannot be inverted).
    #[error("Singular matrix: {0}")]
    SingularMatrix(String),

    /// A matrix is not positive definite (e.g., Cholesky failed).
    #[error("Not positive definite: {0}")]
    NotPositiveDefinite(String),

    /// Arithmetic overflow during computation.
    #[error("Arithmetic overflow in {context}: {message}")]
    Overflow {
        context: String,
        message: String,
    },

    /// Arithmetic underflow during computation.
    #[error("Arithmetic underflow in {context}: {message}")]
    Underflow {
        context: String,
        message: String,
    },

    /// Not enough data points to perform the computation.
    #[error("Insufficient data: need at least {required}, got {got}")]
    InsufficientData {
        required: usize,
        got: usize,
    },

    /// A numeric value is outside its allowed range.
    ///
    /// # Fields
    /// - `param`  — name of the parameter (e.g., `"p"`)
    /// - `lo`     — inclusive lower bound
    /// - `hi`     — inclusive upper bound
    /// - `actual` — the value that was supplied
    #[error("{param} must be in [{lo}, {hi}], got {actual}")]
    OutOfBounds {
        param: String,
        lo: f64,
        hi: f64,
        actual: f64,
    },

    /// A numerical computation produced NaN or infinity unexpectedly.
    #[error("Numerical error in {context}: {message}")]
    Numerical {
        context: String,
        message: String,
    },

    /// Feature or algorithm is not yet implemented.
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Catch-all for errors from external libraries after conversion.
    #[error("External library error: {0}")]
    External(String),
}

/// Convenience alias used by every crate in the workspace.
pub type Result<T> = std::result::Result<T, StatsError>;

// ── Convenience constructors ──────────────────────────────────────────────────
//
// These reduce boilerplate at call sites.
// Instead of:
//   Err(StatsError::Domain(format!("sigma must be positive, got {sigma}")))
// Write:
//   Err(StatsError::domain(format!("sigma must be positive, got {sigma}")))
//
// The difference is ergonomic: `domain()` is shorter and avoids
// confusion with the enum variant in match arms.

impl StatsError {
    /// Create a [`StatsError::Domain`] with a formatted message.
    pub fn domain(msg: impl Into<String>) -> Self {
        Self::Domain(msg.into())
    }

    /// Create a [`StatsError::DimensionMismatch`] with a formatted message.
    pub fn dim_mismatch(msg: impl Into<String>) -> Self {
        Self::DimensionMismatch(msg.into())
    }

    /// Create a [`StatsError::Convergence`] error.
    pub fn convergence(iterations: usize, msg: impl Into<String>) -> Self {
        Self::Convergence {
            iterations,
            message: msg.into(),
        }
    }

    /// Create a [`StatsError::InsufficientData`] error.
    pub fn insufficient_data(required: usize, got: usize) -> Self {
        Self::InsufficientData { required, got }
    }

    /// Create a [`StatsError::OutOfBounds`] error.
    pub fn out_of_bounds(
        param: impl Into<String>,
        lo: f64,
        hi: f64,
        actual: f64,
    ) -> Self {
        Self::OutOfBounds {
            param: param.into(),
            lo,
            hi,
            actual,
        }
    }

    /// Create a [`StatsError::Numerical`] error.
    pub fn numerical(context: impl Into<String>, msg: impl Into<String>) -> Self {
        Self::Numerical {
            context: context.into(),
            message: msg.into(),
        }
    }

    /// Create a [`StatsError::NotImplemented`] error.
    pub fn not_implemented(msg: impl Into<String>) -> Self {
        Self::NotImplemented(msg.into())
    }

    /// Create a [`StatsError::SingularMatrix`] error.
    pub fn singular(msg: impl Into<String>) -> Self {
        Self::SingularMatrix(msg.into())
    }

    /// Create a [`StatsError::NotPositiveDefinite`] error.
    pub fn not_positive_definite(msg: impl Into<String>) -> Self {
        Self::NotPositiveDefinite(msg.into())
    }
}

// ── Standard From conversions ─────────────────────────────────────────────────

impl From<std::num::ParseFloatError> for StatsError {
    fn from(e: std::num::ParseFloatError) -> Self {
        Self::External(e.to_string())
    }
}

impl From<std::num::ParseIntError> for StatsError {
    fn from(e: std::num::ParseIntError) -> Self {
        Self::External(e.to_string())
    }
}

// ── Validation helpers ────────────────────────────────────────────────────────
//
// These are the most frequently written checks across all crates.
// Centralising them here means a one-line call at every parameter boundary.

/// Require that `slice` has at least `n` elements.
///
/// # Errors
/// Returns [`StatsError::InsufficientData`] if `slice.len() < n`.
pub fn require_min_len(slice: &[f64], n: usize) -> Result<()> {
    if slice.len() < n {
        return Err(StatsError::insufficient_data(n, slice.len()));
    }
    Ok(())
}

/// Require that two slices have the same length.
///
/// # Errors
/// Returns [`StatsError::DimensionMismatch`] if lengths differ.
pub fn require_same_len(a: &[f64], b: &[f64], ctx: &str) -> Result<()> {
    if a.len() != b.len() {
        return Err(StatsError::dim_mismatch(format!(
            "{ctx}: arrays have different lengths ({} vs {})",
            a.len(),
            b.len()
        )));
    }
    Ok(())
}

/// Require that every element of `slice` is finite (no NaN, no Inf).
///
/// # Errors
/// Returns [`StatsError::Domain`] if any element fails the check.
pub fn require_finite(slice: &[f64], ctx: &str) -> Result<()> {
    if slice.iter().any(|x| !x.is_finite()) {
        return Err(StatsError::domain(format!(
            "{ctx}: data contains NaN or infinite values"
        )));
    }
    Ok(())
}

/// Require that `value` lies in the closed interval `[lo, hi]`.
///
/// # Errors
/// Returns [`StatsError::OutOfBounds`] if `value < lo || value > hi`.
pub fn require_in_range(
    value: f64,
    lo: f64,
    hi: f64,
    param: &str,
) -> Result<()> {
    if value < lo || value > hi {
        return Err(StatsError::out_of_bounds(param, lo, hi, value));
    }
    Ok(())
}

/// Require that `value > 0.0`.
///
/// # Errors
/// Returns [`StatsError::Domain`] if `value <= 0.0`.
pub fn require_positive(value: f64, param: &str) -> Result<()> {
    if value <= 0.0 {
        return Err(StatsError::domain(format!(
            "{param} must be positive, got {value}"
        )));
    }
    Ok(())
}

/// Require that `value >= 0.0`.
///
/// # Errors
/// Returns [`StatsError::Domain`] if `value < 0.0`.
pub fn require_non_negative(value: f64, param: &str) -> Result<()> {
    if value < 0.0 {
        return Err(StatsError::domain(format!(
            "{param} must be non-negative, got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_require_min_len_passes() {
        assert!(require_min_len(&[1.0, 2.0], 2).is_ok());
    }

    #[test]
    fn test_require_min_len_fails() {
        let err = require_min_len(&[1.0], 2).unwrap_err();
        assert!(matches!(
            err,
            StatsError::InsufficientData { required: 2, got: 1 }
        ));
    }

    #[test]
    fn test_require_finite_passes() {
        assert!(require_finite(&[1.0, 2.0, 3.0], "test").is_ok());
    }

    #[test]
    fn test_require_finite_fails_nan() {
        assert!(require_finite(&[1.0, f64::NAN], "test").is_err());
    }

    #[test]
    fn test_require_finite_fails_inf() {
        assert!(require_finite(&[f64::INFINITY], "test").is_err());
    }

    #[test]
    fn test_require_positive_passes() {
        assert!(require_positive(0.001, "sigma").is_ok());
    }

    #[test]
    fn test_require_positive_fails_zero() {
        assert!(require_positive(0.0, "sigma").is_err());
    }

    #[test]
    fn test_require_positive_fails_negative() {
        assert!(require_positive(-1.0, "sigma").is_err());
    }

    #[test]
    fn test_out_of_bounds_message() {
        let err = require_in_range(1.5, 0.0, 1.0, "p").unwrap_err();
        assert!(err.to_string().contains("p must be in [0, 1]"));
    }

    #[test]
    fn test_error_display_convergence() {
        let err = StatsError::convergence(100, "Newton step did not shrink");
        assert!(err.to_string().contains("100"));
    }
}