use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum StatsError {
    #[error("Domain error: {0}")]
    DomainError(String),

    #[error("Convergence failure: {0}")]
    ConvergenceError(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Singular matrix")]
    SingularMatrix,

    #[error("Not positive definite")]
    NotPositiveDefinite,

    #[error("Computation overflow in {0}")]
    OverflowError(String),

    #[error("Numerical underflow in {0}")]
    UnderflowError(String),

    #[error("Algorithm did not converge after {iterations} iterations")]
    MaxIterationsExceeded { iterations: usize },

    #[error("Insufficient data: need at least {required} observations, got {got}")]
    InsufficientData { required: usize, got: usize },

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Value out of bounds: {param} must be in [{min}, {max}], got {actual}")]
    ValueOutOfBounds {
        param: String,
        min: f64,
        max: f64,
        actual: f64,
    },
}

pub type Result<T> = std::result::Result<T, StatsError>;