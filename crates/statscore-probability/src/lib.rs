//! # `statscore-probability`
//!
//! Probability theory primitives: moments, inequalities, and distribution
//! transforms. Sits between [`statscore-special`] (special functions) and
//! [`statscore-distributions`] (concrete distributions).
//!
//! ## Planned modules
//! - `moments` — raw/central moments, MGF, characteristic functions
//! - `inequalities` — Chebyshev, Markov, Hoeffding, Jensen
//! - `transforms` — probability integral transform, copulas, convolutions
//!
//! ## Dependencies
//! - [`statscore-common`] — traits and errors
//! - [`statscore-special`] — gamma, beta, erf for transform math
//! - [`statscore-linalg`] — linear algebra for multivariate transforms
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for planned modules and status.
//!
//! ## Status
//! Scaffold crate — implementation pending (Phase 0).
//!
//! [`statscore-special`]: https://docs.rs/statscore-special
//! [`statscore-distributions`]: https://docs.rs/statscore-distributions
//! [`statscore-common`]: https://docs.rs/statscore-common
//! [`statscore-linalg`]: https://docs.rs/statscore-linalg

#![warn(missing_docs)]
#![forbid(unsafe_code)]
