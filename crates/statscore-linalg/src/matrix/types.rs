//! Owned matrix and vector types backed by nalgebra.

use std::fmt;

use nalgebra::{DMatrix as NaDMatrix, DVector as NaDVector};
use statscore_common::{Result, StatsError};

/// A dense `m × n` matrix stored in column-major order (nalgebra convention).
#[derive(Clone, PartialEq)]
pub struct DenseMatrix {
    pub(crate) inner: NaDMatrix<f64>,
}

/// A square `n × n` matrix.
#[derive(Clone, PartialEq)]
pub struct SquareMatrix {
    pub(crate) inner: NaDMatrix<f64>,
}

/// A column vector of length `n`.
#[derive(Clone, PartialEq)]
pub struct Vector {
    pub(crate) inner: NaDVector<f64>,
}

impl DenseMatrix {
    /// Number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    /// Number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    /// Element at `(row, col)`.
    ///
    /// # Panics
    /// Panics if the index is out of bounds (debug builds only via nalgebra).
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.inner[(row, col)]
    }

    /// Borrow the underlying nalgebra matrix.
    #[must_use]
    pub fn as_inner(&self) -> &NaDMatrix<f64> {
        &self.inner
    }

    /// Consume and return the underlying nalgebra matrix.
    #[must_use]
    pub fn into_inner(self) -> NaDMatrix<f64> {
        self.inner
    }

    /// Flatten in row-major order (for ndarray / NumPy interop).
    #[must_use]
    pub fn as_row_slice(&self) -> Vec<f64> {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let mut out = Vec::with_capacity(nrows * ncols);
        for r in 0..nrows {
            for c in 0..ncols {
                out.push(self.inner[(r, c)]);
            }
        }
        out
    }

    pub(crate) fn from_inner(inner: NaDMatrix<f64>) -> Self {
        Self { inner }
    }
}

impl SquareMatrix {
    /// Matrix dimension `n` (rows = columns).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.inner.nrows()
    }

    /// Element at `(row, col)`.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.inner[(row, col)]
    }

    /// Borrow the underlying nalgebra matrix.
    #[must_use]
    pub fn as_inner(&self) -> &NaDMatrix<f64> {
        &self.inner
    }

    /// Consume and return the underlying nalgebra matrix.
    #[must_use]
    pub fn into_inner(self) -> NaDMatrix<f64> {
        self.inner
    }

    /// View as a general dense matrix.
    #[must_use]
    pub fn as_dense(&self) -> DenseMatrix {
        DenseMatrix::from_inner(self.inner.clone())
    }

    /// Flatten in row-major order.
    #[must_use]
    pub fn as_row_slice(&self) -> Vec<f64> {
        self.as_dense().as_row_slice()
    }

    pub(crate) fn from_inner(inner: NaDMatrix<f64>) -> Result<Self> {
        if inner.nrows() != inner.ncols() {
            return Err(StatsError::dim_mismatch(format!(
                "expected square matrix, got {}×{}",
                inner.nrows(),
                inner.ncols()
            )));
        }
        Ok(Self { inner })
    }

    pub(crate) fn from_inner_unchecked(inner: NaDMatrix<f64>) -> Self {
        Self { inner }
    }
}

impl Vector {
    /// Vector length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Element at index `i`.
    #[must_use]
    pub fn get(&self, index: usize) -> f64 {
        self.inner[index]
    }

    /// Borrow the underlying nalgebra vector.
    #[must_use]
    pub fn as_inner(&self) -> &NaDVector<f64> {
        &self.inner
    }

    /// Consume and return the underlying nalgebra vector.
    #[must_use]
    pub fn into_inner(self) -> NaDVector<f64> {
        self.inner
    }

    /// Copy elements into a `Vec<f64>`.
    #[must_use]
    pub fn as_slice(&self) -> Vec<f64> {
        self.inner.as_slice().to_vec()
    }

    pub(crate) fn from_inner(inner: NaDVector<f64>) -> Self {
        Self { inner }
    }
}

impl fmt::Debug for DenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DenseMatrix({}×{}, row-major = {:?})",
            self.nrows(),
            self.ncols(),
            self.as_row_slice()
        )
    }
}

impl fmt::Debug for SquareMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SquareMatrix({}×{}, row-major = {:?})",
            self.dim(),
            self.dim(),
            self.as_row_slice()
        )
    }
}

impl fmt::Debug for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector(len = {}, data = {:?})", self.len(), self.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_matrix_rejects_non_square() {
        let m = NaDMatrix::zeros(2, 3);
        assert!(SquareMatrix::from_inner(m).is_err());
    }
}
