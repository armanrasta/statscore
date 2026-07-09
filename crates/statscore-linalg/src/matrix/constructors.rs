//! Matrix and vector constructors.

use nalgebra::{DMatrix as NaDMatrix, DVector as NaDVector};
use statscore_common::{Result, StatsError};

use super::types::{DenseMatrix, SquareMatrix, Vector};

/// Create an `m × n` zero matrix.
///
/// # Example
/// ```
/// use statscore_linalg::matrix::{zeros, DenseMatrix};
/// let m = zeros(2, 3);
/// assert_eq!(m.nrows(), 2);
/// assert_eq!(m.ncols(), 3);
/// ```
#[must_use]
pub fn zeros(rows: usize, cols: usize) -> DenseMatrix {
    DenseMatrix::from_inner(NaDMatrix::zeros(rows, cols))
}

/// Create an `m × n` matrix filled with ones.
#[must_use]
pub fn ones(rows: usize, cols: usize) -> DenseMatrix {
    DenseMatrix::from_inner(NaDMatrix::from_element(rows, cols, 1.0))
}

/// Create an `n × n` identity matrix.
///
/// # Example
/// ```
/// use statscore_linalg::matrix::{identity, SquareMatrix};
/// let i = identity(3);
/// assert_eq!(i.dim(), 3);
/// assert!((i.get(0, 0) - 1.0).abs() < 1e-15);
/// assert!((i.get(0, 1)).abs() < 1e-15);
/// ```
#[must_use]
pub fn identity(n: usize) -> SquareMatrix {
    SquareMatrix::from_inner_unchecked(NaDMatrix::identity(n, n))
}

/// Build an `m × n` matrix from a row-major slice.
///
/// # Errors
/// Returns [`StatsError::DimensionMismatch`] if `data.len() != rows * cols`.
///
/// # Example
/// ```
/// use statscore_linalg::matrix::from_row_slice;
/// let m = from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
/// assert_eq!(m.get(0, 0), 1.0);
/// assert_eq!(m.get(1, 1), 4.0);
/// ```
pub fn from_row_slice(rows: usize, cols: usize, data: &[f64]) -> Result<DenseMatrix> {
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| StatsError::dim_mismatch("matrix dimensions overflow usize"))?;
    if data.len() != expected {
        return Err(StatsError::dim_mismatch(format!(
            "expected {} elements for a {rows}×{cols} matrix, got {}",
            expected,
            data.len()
        )));
    }
    Ok(DenseMatrix::from_inner(NaDMatrix::from_row_slice(
        rows, cols, data,
    )))
}

/// Build an `n × n` square matrix from a row-major slice.
///
/// # Errors
/// Returns [`StatsError::DimensionMismatch`] if `data.len() != n²`.
pub fn square_from_row_slice(n: usize, data: &[f64]) -> Result<SquareMatrix> {
    from_row_slice(n, n, data).map(|m| SquareMatrix::from_inner_unchecked(m.into_inner()))
}

/// Build a column vector from a slice.
///
/// # Errors
/// Returns [`StatsError::DimensionMismatch`] if the slice is empty when required
/// by the caller (this function accepts any length ≥ 0).
pub fn vector_from_slice(data: &[f64]) -> Vector {
    Vector::from_inner(NaDVector::from_row_slice(data))
}

/// Build a column vector of given length filled with a constant.
#[must_use]
pub fn column_vector(len: usize, value: f64) -> Vector {
    Vector::from_inner(NaDVector::from_element(len, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_diagonal() {
        let i = identity(4);
        for r in 0..4 {
            for c in 0..4 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!((i.get(r, c) - expected).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn from_row_slice_rejects_bad_length() {
        assert!(from_row_slice(2, 2, &[1.0, 2.0, 3.0]).is_err());
    }
}
