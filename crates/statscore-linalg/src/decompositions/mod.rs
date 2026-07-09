//! Matrix decompositions.

mod cholesky;
mod eigen;
mod qr;
mod svd;

pub use cholesky::{cholesky, CholeskyDecomposition};
pub use eigen::{eigen_symmetric, EigenDecomposition};
pub use qr::{qr, QrDecomposition};
pub use svd::{svd, SvdDecomposition};
