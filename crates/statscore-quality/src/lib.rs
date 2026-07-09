//! # `statscore-quality`
//!
//! Statistical process control: Shewhart control charts, CUSUM, EWMA, and
//! process capability indices.
//!
//! ## Planned modules
//! - `control_charts` — X̄-R, X̄-S, p, c, u charts, CUSUM, EWMA
//! - `capability` — Cp, Cpk, Pp, Ppk, Cpm
//!
//! ## Dependencies
//! - [`statscore-common`] — types and errors
//! - [`statscore-distributions`] — control limit distributions
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for planned modules and status.
//!
//! ## Status
//! Scaffold crate — implementation pending (Phase 3).
//!
//! [`statscore-common`]: https://docs.rs/statscore-common

#![warn(missing_docs)]
#![forbid(unsafe_code)]
