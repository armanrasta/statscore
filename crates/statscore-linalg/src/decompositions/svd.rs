//! Singular value decomposition.

use statscore_common::Result;

use crate::matrix::DenseMatrix;

/// Result of an economic SVD `A = U Σ Vᵀ`.
#[derive(Debug, Clone, PartialEq)]
pub struct SvdDecomposition {
    /// Left singular vectors (`m × k`).
    pub u: DenseMatrix,
    /// Singular values in descending order (length `k`).
    pub singular_values: Vec<f64>,
    /// Right singular vectors transposed (`k × n`).
    pub v_t: DenseMatrix,
}

/// Compute the thin SVD of `matrix`.
///
/// # Example
/// ```
/// use statscore_linalg::decompositions::svd;
/// use statscore_linalg::matrix::from_row_slice;
///
/// let a = from_row_slice(2, 2, &[3.0, 1.0, 1.0, 3.0]).unwrap();
/// let svd = svd(&a).unwrap();
/// assert_eq!(svd.singular_values.len(), 2);
/// ```
pub fn svd(matrix: &DenseMatrix) -> Result<SvdDecomposition> {
    let decomp = matrix.as_inner().svd(true, true);
    let u = decomp
        .u
        .ok_or_else(|| statscore_common::StatsError::numerical("svd", "U factor missing"))?;
    let v_t = decomp
        .v_t
        .ok_or_else(|| statscore_common::StatsError::numerical("svd", "Vᵀ factor missing"))?;
    Ok(SvdDecomposition {
        u: DenseMatrix::from_inner(u),
        singular_values: decomp.singular_values.as_slice().to_vec(),
        v_t: DenseMatrix::from_inner(v_t),
    })
}

impl SvdDecomposition {
    /// Reconstruct `A = U Σ Vᵀ`.
    #[must_use]
    pub fn reconstruct(&self) -> DenseMatrix {
        let sigma = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(
            &self.singular_values,
        ));
        DenseMatrix::from_inner(self.u.as_inner() * sigma * self.v_t.as_inner())
    }

    /// Numerical rank using a relative tolerance.
    #[must_use]
    pub fn rank(&self, tol: f64) -> usize {
        if self.singular_values.is_empty() {
            return 0;
        }
        let max_sv = self.singular_values[0];
        if max_sv == 0.0 {
            return 0;
        }
        self.singular_values
            .iter()
            .filter(|&&s| s / max_sv > tol)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::from_row_slice;
    use approx::assert_relative_eq;

    #[test]
    fn svd_reconstructs() {
        let a = from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let decomp = svd(&a).unwrap();
        let reconstructed = decomp.reconstruct();
        for r in 0..3 {
            for c in 0..2 {
                assert_relative_eq!(
                    reconstructed.get(r, c),
                    a.get(r, c),
                    epsilon = 1e-9
                );
            }
        }
    }

    #[test]
    fn svd_singular_values_positive() {
        let a = from_row_slice(2, 2, &[3.0, 1.0, 1.0, 3.0]).unwrap();
        let decomp = svd(&a).unwrap();
        for &s in &decomp.singular_values {
            assert!(s >= 0.0);
        }
    }
}
