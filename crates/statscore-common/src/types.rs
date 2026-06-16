//! Shared numeric type aliases used across the entire workspace.
//!
//! Every crate imports these instead of writing `ndarray::Array1<f64>` directly,
//! so if we ever go generic over float type, we change one file.

use ndarray::{Array1,Array2, ArrayView1, ArrayView2};

///! The floating-point scalar type used throughout the library.
///!
///! Currently `f64`. All statistical computations run in double precision.
pub type Scalar = f64;

/// A 1-dimensional owned array of [`Scalar`] values.
///
/// Represents vectors, samples, time series, etc.
pub type Vector = Array1<Scalar>;

/// A 2-dimensional owned array of [`Scalar`] values, stored in row-major order.
///
/// Represents matrices, data tables (rows = observations, columns = features), etc.
pub type Matrix = Array2<Scalar>;

/// A borrowed view of a 1-dimensional array.
///
/// Used for zero-copy function arguments.
pub type VectorView<'a> = ArrayView1<'a, Scalar>;

/// A borrowed view of a 2-dimensional array.
///
/// Used for zero-copy function arguments.
pub type MatrixView<'a> = ArrayView2<'a, Scalar>;
