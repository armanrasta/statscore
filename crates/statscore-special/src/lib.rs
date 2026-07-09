//! # `statscore-special`
//!
//! Special mathematical functions that form the numerical bedrock of the
//! `statscore` workspace: the gamma family, the beta family, the error
//! function, modified Bessel functions, and log-space combinatorics.
//!
//! Every function operates in `f64` and returns `f64::NAN` (or `±∞` where that
//! is the correct mathematical limit) for out-of-domain inputs, matching the
//! SciPy/cephes convention. This keeps the special-function layer free of
//! `Result` noise; higher-level crates validate parameters at their own
//! boundaries.
//!
//! ## Modules
//! - [`gamma`] — `ln_gamma`, `gamma`, `digamma`, `trigamma`, regularized
//!   incomplete gamma (`gammainc` / `gammaincc`)
//! - [`beta`] — `ln_beta`, `beta`, regularized incomplete beta (`betainc`) and
//!   its inverse (`betaincinv`)
//! - [`erf`] — `erf`, `erfc`, `erf_inv`, `erfc_inv`
//! - [`bessel`] — modified Bessel `i0`, `i1`, `k0`, `k1` (plus scaled variants
//!   and `ln_i0`)
//! - [`combinatorics`] — `ln_factorial`, `factorial`, `ln_choose`, `choose`,
//!   `ln_perm`
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for module overview, accuracy targets,
//! and examples.
//! ```
//! use statscore_special::gamma::gamma;
//! use statscore_special::erf::erf;
//!
//! assert!((gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-12);
//! assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-14);
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod bessel;
pub mod beta;
pub mod combinatorics;
pub mod erf;
pub mod gamma;

// ── Flat re-exports for ergonomic imports ────────────────────────────────────

pub use gamma::{EULER_GAMMA, digamma, gamma, gammainc, gammaincc, ln_gamma, trigamma};

pub use beta::{beta, betainc, betaincinv, ln_beta};

pub use erf::{erf, erf_inv, erfc, erfc_inv};

pub use bessel::{i0, i0e, i1, i1e, k0, k0e, k1, k1e, ln_i0};

pub use combinatorics::{choose, factorial, ln_choose, ln_factorial, ln_perm};
