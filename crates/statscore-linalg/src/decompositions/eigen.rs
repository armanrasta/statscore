//! Symmetric eigendecomposition.

use statscore_common::{Result, StatsError};

use crate::matrix::{DenseMatrix, SquareMatrix};

/// Eigendecomposition of a real symmetric matrix: `A = Q Λ Qᵀ`.
#[derive(Debug, Clone, PartialEq)]
pub struct EigenDecomposition {
    /// Eigenvalues in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Eigenvector matrix (columns are eigenvectors).
    pub eigenvectors: DenseMatrix,
}

/// Compute the eigendecomposition of a real symmetric matrix.
///
/// The input is treated as symmetric; only the upper triangle is used by
/// nalgebra's `symmetric_eigen`.
///
/// # Example
/// ```
/// use statscore_linalg::decompositions::eigen_symmetric;
/// use statscore_linalg::matrix::square_from_row_slice;
///
/// let a = square_from_row_slice(2, &[2.0, 1.0, 1.0, 2.0]).unwrap();
/// let eig = eigen_symmetric(&a).unwrap();
/// assert_eq!(eig.eigenvalues.len(), 2);
/// ```
pub fn eigen_symmetric(matrix: &SquareMatrix) -> Result<EigenDecomposition> {
    if matrix.dim() == 0 {
        return Err(StatsError::domain("matrix dimension must be positive"));
    }
    let decomp = matrix.as_inner().symmetric_eigen();
    Ok(EigenDecomposition {
        eigenvalues: decomp.eigenvalues.as_slice().to_vec(),
        eigenvectors: DenseMatrix::from_inner(decomp.eigenvectors),
    })
}

impl EigenDecomposition {
    /// Reconstruct `A = Q Λ Qᵀ`.
    #[must_use]
    pub fn reconstruct(&self) -> SquareMatrix {
        let lambda = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(
            &self.eigenvalues,
        ));
        let q = self.eigenvectors.as_inner();
        SquareMatrix::from_inner_unchecked(&(q * lambda * q.transpose()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::square_from_row_slice;
    use approx::assert_relative_eq;

    #[test]
    fn eigen_symmetric_reconstructs() {
        let a = square_from_row_slice(2, &[2.0, 1.0, 1.0, 2.0]).unwrap();
        let eig = eigen_symmetric(&a).unwrap();
        let reconstructed = eig.reconstruct();
        for r in 0..2 {
            for c in 0..2 {
                assert_relative_eq!(reconstructed.get(r, c), a.get(r, c), epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn eigenvalues_of_identity() {
        let a = crate::matrix::identity(3);
        let eig = eigen_symmetric(&a).unwrap();
        for &lambda in &eig.eigenvalues {
            assert_relative_eq!(lambda, 1.0, epsilon = 1e-10);
        }
    }
}
