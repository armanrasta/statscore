//! QR decomposition.

use statscore_common::Result;

use crate::matrix::DenseMatrix;

/// Result of a QR factorization `A = Q R`.
#[derive(Debug, Clone, PartialEq)]
pub struct QrDecomposition {
    /// Orthogonal factor `Q` (`m × min(m, n)` when thin).
    pub q: DenseMatrix,
    /// Upper-triangular factor `R` (`min(m, n) × n`).
    pub r: DenseMatrix,
}

/// Compute the thin QR decomposition of `matrix`.
///
/// # Example
/// ```
/// use statscore_linalg::decompositions::qr;
/// use statscore_linalg::matrix::from_row_slice;
///
/// let a = from_row_slice(3, 2, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
/// let qr = qr(&a).unwrap();
/// assert_eq!(qr.q.nrows(), 3);
/// assert_eq!(qr.r.ncols(), 2);
/// ```
pub fn qr(matrix: &DenseMatrix) -> Result<QrDecomposition> {
    let decomp = matrix.as_inner().clone().qr();
    Ok(QrDecomposition {
        q: DenseMatrix::from_inner(decomp.q()),
        r: DenseMatrix::from_inner(decomp.r()),
    })
}

impl QrDecomposition {
    /// Reconstruct `A ≈ Q R` (exact up to floating-point error).
    #[must_use]
    pub fn reconstruct(&self) -> DenseMatrix {
        DenseMatrix::from_inner(self.q.as_inner() * self.r.as_inner())
    }

    /// Solve the least-squares problem `min ||A x - b||₂` when `A` is full column rank.
    ///
    /// # Errors
    /// Returns [`statscore_common::StatsError::SingularMatrix`] if `R` is rank-deficient.
    pub fn solve_least_squares(&self, b: &crate::matrix::Vector) -> Result<crate::matrix::Vector> {
        if b.len() != self.q.nrows() {
            return Err(statscore_common::StatsError::dim_mismatch(format!(
                "expected rhs length {}, got {}",
                self.q.nrows(),
                b.len()
            )));
        }
        // Thin QR: x = R⁻¹ Qᵀ b
        let qt_b = self.q.as_inner().transpose() * b.as_inner();
        let x = self
            .r
            .as_inner()
            .clone()
            .solve_upper_triangular(&qt_b)
            .ok_or_else(|| {
                statscore_common::StatsError::singular("QR least-squares solve failed")
            })?;
        Ok(crate::matrix::Vector::from_inner(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::{from_row_slice, vector_from_slice};
    use approx::assert_relative_eq;

    #[test]
    fn qr_reconstructs() {
        let a = from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let decomp = qr(&a).unwrap();
        let reconstructed = decomp.reconstruct();
        for r in 0..3 {
            for c in 0..2 {
                assert_relative_eq!(reconstructed.get(r, c), a.get(r, c), epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn qr_least_squares_overdetermined() {
        // Fit y ≈ a + b x for points (0,1), (1,2), (2,2)
        let a = from_row_slice(3, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0]).unwrap();
        let b = vector_from_slice(&[1.0, 2.0, 2.0]);
        let decomp = qr(&a).unwrap();
        let x = decomp.solve_least_squares(&b).unwrap();
        assert!(x.len() == 2);
        assert!(x.get(0).is_finite());
        assert!(x.get(1).is_finite());
    }
}
