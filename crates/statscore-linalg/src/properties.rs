//! Matrix properties: trace, determinant, rank, condition number, pseudoinverse.

use statscore_common::{Result, StatsError};

use crate::decompositions::svd;
use crate::matrix::{DenseMatrix, SquareMatrix};

/// Sum of the diagonal elements.
///
/// # Example
/// ```
/// use statscore_linalg::matrix::identity;
/// use statscore_linalg::properties::trace;
///
/// assert!((trace(&identity(3)) - 3.0).abs() < 1e-15);
/// ```
#[must_use]
pub fn trace(m: &SquareMatrix) -> f64 {
    m.as_inner().trace()
}

/// Matrix determinant.
///
/// # Example
/// ```
/// use statscore_linalg::matrix::square_from_row_slice;
/// use statscore_linalg::properties::det;
///
/// let m = square_from_row_slice(2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
/// assert!((det(&m) - (-2.0)).abs() < 1e-12);
/// ```
#[must_use]
pub fn det(m: &SquareMatrix) -> f64 {
    m.as_inner().determinant()
}

/// Numerical rank via SVD: count of singular values above `tol · σ_max`.
///
/// # Example
/// ```
/// use statscore_linalg::matrix::from_row_slice;
/// use statscore_linalg::properties::rank;
///
/// let m = from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]).unwrap();
/// assert_eq!(rank(&m, 1e-10).unwrap(), 1);
/// ```
pub fn rank(m: &DenseMatrix, tol: f64) -> Result<usize> {
    Ok(svd(m)?.rank(tol))
}

/// Condition number κ(A) = σ_max / σ_min (2-norm).
///
/// # Errors
/// Returns [`StatsError::SingularMatrix`] if `σ_min ≈ 0`.
pub fn condition_number(m: &SquareMatrix) -> Result<f64> {
    let decomp = svd(&m.as_dense())?;
    let s = &decomp.singular_values;
    if s.is_empty() {
        return Err(StatsError::singular("empty matrix"));
    }
    let s_max = s[0];
    let s_min = s.last().copied().unwrap_or(0.0);
    if s_max == 0.0 || s_min.abs() < f64::EPSILON {
        return Err(StatsError::singular("matrix is singular"));
    }
    Ok(s_max / s_min)
}

/// Moore–Penrose pseudoinverse A⁺ via thin SVD.
///
/// Singular values below `tol · σ_max` are zeroed before inversion.
///
/// # Example
/// ```
/// use statscore_linalg::matrix::from_row_slice;
/// use statscore_linalg::properties::pinv;
///
/// let m = from_row_slice(2, 1, &[1.0, 2.0]).unwrap();
/// let mp = pinv(&m, 1e-12).unwrap();
/// assert_eq!(mp.nrows(), 1);
/// assert_eq!(mp.ncols(), 2);
/// ```
pub fn pinv(m: &DenseMatrix, tol: f64) -> Result<DenseMatrix> {
    let decomp = svd(m)?;
    let max_sv = decomp.singular_values.first().copied().unwrap_or(0.0);
    let inv_diag: Vec<f64> = decomp
        .singular_values
        .iter()
        .map(|&s| {
            if max_sv > 0.0 && s / max_sv > tol {
                1.0 / s
            } else {
                0.0
            }
        })
        .collect();
    let sigma_inv = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_vec(inv_diag));
    // A⁺ = V Σ⁺ Uᵀ  where  v_t = Vᵀ
    let v = decomp.v_t.as_inner().transpose();
    let ut = decomp.u.as_inner().transpose();
    Ok(DenseMatrix::from_inner(v * sigma_inv * ut))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::{from_row_slice, identity, square_from_row_slice};
    use approx::assert_relative_eq;

    #[test]
    fn trace_identity() {
        assert_relative_eq!(trace(&identity(4)), 4.0, epsilon = 1e-15);
    }

    #[test]
    fn det_2x2() {
        let m = square_from_row_slice(2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_relative_eq!(det(&m), -2.0, epsilon = 1e-12);
    }

    #[test]
    fn rank_deficient() {
        let m = from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]).unwrap();
        assert_eq!(rank(&m, 1e-10).unwrap(), 1);
    }

    #[test]
    fn condition_number_identity() {
        assert_relative_eq!(
            condition_number(&identity(3)).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn pinv_overdetermined() {
        let m = from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let mp = pinv(&m, 1e-12).unwrap();
        // A A⁺ A ≈ A
        let a_pinv_a = m.as_inner() * mp.as_inner() * m.as_inner();
        for r in 0..3 {
            for c in 0..2 {
                assert_relative_eq!(a_pinv_a[(r, c)], m.as_inner()[(r, c)], epsilon = 1e-9);
            }
        }
    }
}
