#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod decompositions;
pub mod matrix;
pub mod properties;
pub mod solve;

// flat re-exports
pub use decompositions::{cholesky, eigen_symmetric, qr, svd, ...};
pub use matrix::{DenseMatrix, SquareMatrix, Vector, identity, zeros, ...};
pub use properties::{trace, det, rank, condition_number, pinv};
pub use solve::{solve_linear_system, solve_least_squares};