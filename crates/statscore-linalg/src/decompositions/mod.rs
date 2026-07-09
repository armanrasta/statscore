//! Matrix decompositions.

mod cholesky;
mod eigen;
mod qr;
mod svd;

pub use cholesky::{CholeskyDecomposition, cholesky};
pub use eigen::{EigenDecomposition, eigen_symmetric};
pub use qr::{QrDecomposition, qr};
pub use svd::{SvdDecomposition, svd};
