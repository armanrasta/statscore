//! # `statscore-multivariate`
//!
//! Multivariate analysis: dimensionality reduction, clustering, classification,
//! and MANOVA.
//!
//! ## Planned modules
//! - `reduction` — PCA (SVD), factor analysis, classical MDS
//! - `clustering` — K-means, hierarchical, DBSCAN
//! - `classification` — LDA, QDA
//! - `manova` — multivariate analysis of variance
//!
//! ## Dependencies
//! - [`statscore-common`] — types and errors
//! - [`statscore-linalg`] — SVD, eigendecomposition
//! - [`statscore-distributions`] — multivariate normal
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for planned modules and status.
//!
//! ## Status
//! Scaffold crate — implementation pending (Phase 2).
//!
//! [`statscore-common`]: https://docs.rs/statscore-common

#![warn(missing_docs)]
#![forbid(unsafe_code)]
