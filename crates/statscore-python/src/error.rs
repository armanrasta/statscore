//! Map [`StatsError`] into Python exceptions.

use pyo3::PyErr;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use statscore_common::StatsError;

/// Convert a [`StatsError`] into a Python exception.
pub fn stats_to_py(err: StatsError) -> PyErr {
    match err {
        StatsError::Domain(_)
        | StatsError::OutOfBounds { .. }
        | StatsError::DimensionMismatch(_)
        | StatsError::NotPositiveDefinite(_)
        | StatsError::SingularMatrix(_)
        | StatsError::InsufficientData { .. } => PyValueError::new_err(err.to_string()),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}
