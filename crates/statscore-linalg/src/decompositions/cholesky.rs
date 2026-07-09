//! Cholesky decomposition for positive-definite matrices.

use statscore_common::{Result, StatsError};

use crate::matrix::{SquareMatrix, Vector};

/// Result of a successful Cholesky factorization `A = L Lᵀ`.
#[derive(Debug, Clone, PartialEq)]
pub struct CholeskyDecomposition {
    /// Lower-triangular factor with positive diagonal.
    pub l: SquareMatrix,
}

/// Compute the Cholesky decomposition of a symmetric positive-definite matrix.
///
/// # Errors
/// Returns [`StatsError::NotPositiveDefinite`] if the matrix is not PD.
///
/// # Example
/// ```
/// use statscore_linalg::decompositions::cholesky;
/// use statscore_linalg::matrix::{identity, SquareMatrix};
///
/// let a = identity(3);
/// let chol = cholesky(&a).unwrap();
/// assert!((chol.l.get(0, 0) - 1.0).abs() < 1e-12);
/// ```
pub fn cholesky(matrix: &SquareMatrix) -> Result<CholeskyDecomposition> {
    let factor = matrix
        .as_inner()
        .cholesky()
        .ok_or_else(|| StatsError::not_positive_definite("Cholesky decomposition failed"))?;
    Ok(CholeskyDecomposition {
        l: SquareMatrix::from_inner_unchecked(factor.l().clone()),
    })
}

impl CholeskyDecomposition {
    /// Solve `A x = b` using the Cholesky factors.
    ///
    /// # Errors
    /// Returns [`StatsError::DimensionMismatch`] if `b.len() != A.dim()`.
    pub fn solve(&self, b: &Vector) -> Result<Vector> {
        let n = self.l.dim();
        if b.len() != n {
            return Err(StatsError::dim_mismatch(format!(
                "expected rhs length {n}, got {}",
                b.len()
            )));
        }
        // The L factor is already lower-triangular from a valid Cholesky.
        // Use forward and backward solve directly.
        let l = self.l.as_inner();
        // Solve L y = b (forward substitution)
        let y = l
            .clone()
            .lower_triangle()
            .solve_lower_triangular(b.as_inner())
            .map_err(|_| StatsError::singular("Cholesky forward solve failed"))?;
        // Solve Lᵗ x = y (backward substitution)
        let x = l
            .transpose()
            .lower_triangle()
            .solve_lower_triangular(&y)
            .map_err(|_| StatsError::singular("Cholesky backward solve failed"))?;
        Ok(Vector::from_inner(x))
    }

    /// Reconstruct `A = L Lᵀ`.
    #[must_use]
    pub fn reconstruct(&self) -> SquareMatrix {
        let l = self.l.as_inner();
        SquareMatrix::from_inner_unchecked(&(l * l.transpose()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::square_from_row_slice;
    use approx::assert_relative_eq;

    #[test]
    fn cholesky_roundtrip() {
        // Symmetric PD matrix [[4, 2], [2, 3]]
        let a = square_from_row_slice(2, &[4.0, 2.0, 2.0, 3.0]).unwrap();
        let chol = cholesky(&a).unwrap();
        let reconstructed = chol.reconstruct();
        for r in 0..2 {
            for c in 0..2 {
                assert_relative_eq!(reconstructed.get(r, c), a.get(r, c), epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn cholesky_solve() {
        let a = square_from_row_slice(2, &[4.0, 2.0, 2.0, 3.0]).unwrap();
        let b = Vector::from_inner(nalgebra::DVector::from_row_slice(&[1.0, 2.0]));
        let chol = cholesky(&a).unwrap();
        let x = chol.solve(&b).unwrap();
        // A x should equal b
        let ax = a.as_inner() * x.as_inner();
        for i in 0..2 {
            assert_relative_eq!(ax[i], b.get(i), epsilon = 1e-10);
        }
    }

    #[test]
    fn cholesky_rejects_indefinite() {
        let a = square_from_row_slice(2, &[1.0, 2.0, 2.0, 1.0]).unwrap();
        assert!(cholesky(&a).is_err());
    }
}
