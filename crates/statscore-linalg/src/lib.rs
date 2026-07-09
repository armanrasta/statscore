//! # `statscore-linalg`
//!
//! Statistical linear algebra: matrix types, decompositions, solvers, and matrix
//! properties. Backed by pure-Rust [`nalgebra`] — no system BLAS required.
//!
//! ## Modules
//! - [`matrix`] — [`DenseMatrix`], [`SquareMatrix`], [`Vector`] newtypes and constructors
//! - [`decompositions`] — Cholesky, QR, SVD, symmetric eigendecomposition
//! - [`solve`] — linear systems and least-squares
//! - [`properties`] — trace, determinant, rank, condition number, pseudoinverse
//!
//! ## Dependencies
//! - [`statscore-common`] — shared [`StatsError`] type
//! - [`nalgebra`] — pure-Rust matrix decompositions
//!
//! ## Guide
//!
//! Full documentation in [`docs/`](docs/README.md):
//!
//! - [Matrix types](docs/matrix.md)
//! - [Decompositions](docs/decompositions.md)
//! - [Solvers](docs/solve.md)
//! - [Properties](docs/properties.md)
//! - [Error handling](docs/errors.md)
//! - [Statistical examples](docs/examples.md)
//!
//! ## Example
//! ```
//! use statscore_linalg::matrix::{identity, vector_from_slice};
//! use statscore_linalg::solve::solve_linear_system;
//!
//! let a = identity(2);
//! let b = vector_from_slice(&[3.0, 4.0]);
//! let x = solve_linear_system(&a, &b).unwrap();
//! assert!((x.get(0) - 3.0).abs() < 1e-12);
//! ```
//!
//! [`nalgebra`]: https://nalgebra.org
//! [`statscore-common`]: https://docs.rs/statscore-common
//! [`StatsError`]: statscore_common::StatsError

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
