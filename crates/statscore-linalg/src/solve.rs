//! Linear system solvers.

use statscore_common::{Result, StatsError};

use crate::decompositions::cholesky;
use crate::matrix::{DenseMatrix, SquareMatrix, Vector};

/// Solve the square linear system `A x = b`.
///
/// Uses Cholesky factorization when `A` is symmetric positive definite, otherwise
/// falls back to LU decomposition.
///
/// # Errors
/// - [`StatsError::DimensionMismatch`] if shapes are incompatible.
/// - [`StatsError::SingularMatrix`] if `A` is singular.
///
/// # Example
/// ```
/// use statscore_linalg::matrix::{identity, vector_from_slice};
/// use statscore_linalg::solve::solve_linear_system;
///
/// let a = identity(2);
/// let b = vector_from_slice(&[3.0, 4.0]);
/// let x = solve_linear_system(&a, &b).unwrap();
/// assert!((x.get(0) - 3.0).abs() < 1e-12);
/// assert!((x.get(1) - 4.0).abs() < 1e-12);
/// ```
pub fn solve_linear_system(a: &SquareMatrix, b: &Vector) -> Result<Vector> {
    if b.len() != a.dim() {
        return Err(StatsError::dim_mismatch(format!(
            "expected rhs length {}, got {}",
            a.dim(),
            b.len()
        )));
    }

    if let Ok(chol) = cholesky(a) {
        return chol.solve(b);
    }

    let lu = a
        .as_inner()
        .lu()
        .solve(b.as_inner())
        .ok_or_else(|| StatsError::singular("LU solve failed: matrix is singular"))?;
    Ok(Vector::from_inner(lu))
}

/// Solve the overdetermined least-squares problem `min ||A x - b||₂`.
///
/// # Errors
/// Returns [`StatsError::DimensionMismatch`] if `b.len() != A.nrows()`.
pub fn solve_least_squares(a: &DenseMatrix, b: &Vector) -> Result<Vector> {
    if b.len() != a.nrows() {
        return Err(StatsError::dim_mismatch(format!(
            "expected rhs length {}, got {}",
            a.nrows(),
            b.len()
        )));
    }
    let x = a
        .as_inner()
        .qr()
        .solve(b.as_inner())
        .ok_or_else(|| StatsError::singular("least-squares solve failed"))?;
    Ok(Vector::from_inner(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::{from_row_slice, square_from_row_slice};
    use approx::assert_relative_eq;

    #[test]
    fn solve_identity_system() {
        let a = crate::matrix::identity(3);
        let b = Vector::from_inner(nalgebra::DVector::from_row_slice(&[1.0, 2.0, 3.0]));
        let x = solve_linear_system(&a, &b).unwrap();
        for i in 0..3 {
            assert_relative_eq!(x.get(i), b.get(i), epsilon = 1e-12);
        }
    }

    #[test]
    fn solve_spd_system() {
        let a = square_from_row_slice(2, &[4.0, 2.0, 2.0, 3.0]).unwrap();
        let b = Vector::from_inner(nalgebra::DVector::from_row_slice(&[1.0, 2.0]));
        let x = solve_linear_system(&a, &b).unwrap();
        let ax = a.as_inner() * x.as_inner();
        assert_relative_eq!(ax[0], b.get(0), epsilon = 1e-10);
        assert_relative_eq!(ax[1], b.get(1), epsilon = 1e-10);
    }

    #[test]
    fn solve_least_squares_regression_setup() {
        let design = from_row_slice(4, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();
        let y = Vector::from_inner(nalgebra::DVector::from_row_slice(&[1.0, 2.0, 2.0, 3.0]));
        let beta = solve_least_squares(&design, &y).unwrap();
        assert_eq!(beta.len(), 2);
        assert!(beta.get(0).is_finite());
        assert!(beta.get(1).is_finite());
    }
}
