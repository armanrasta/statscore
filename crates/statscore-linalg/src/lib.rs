//! # `statscore-linalg`
//!
//! Statistical linear algebra: matrix types, decompositions, solvers, and
//! matrix properties. Backed by pure-Rust [`nalgebra`] — no system BLAS required.
//!
//! [`nalgebra`]: https://nalgebra.org

#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod decompositions;
pub mod matrix;
pub mod properties;
pub mod solve;

pub use decompositions::{
    CholeskyDecomposition, EigenDecomposition, QrDecomposition, SvdDecomposition, cholesky,
    eigen_symmetric, qr, svd,
};
pub use matrix::{
    DenseMatrix, SquareMatrix, Vector, column_vector, from_row_slice, identity, ones,
    square_from_row_slice, vector_from_slice, zeros,
};
pub use properties::{condition_number, det, pinv, rank, trace};
pub use solve::{solve_least_squares, solve_linear_system};
