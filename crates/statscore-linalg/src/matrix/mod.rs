//! Matrix and vector newtypes with construction invariants.

mod constructors;
mod types;

pub use constructors::{
    column_vector,
    from_row_slice,
    identity,
    ones,
    square_from_row_slice,
    vector_from_slice,
    zeros,
};
pub use types::{DenseMatrix, SquareMatrix, Vector};
