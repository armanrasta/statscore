//! # `statscore-timeseries`
//!
//! Time series analysis and forecasting for the `statscore` workspace:
//! baselines (naive / drift), ETS, ARIMA, Prophet-style additive models,
//! Markov chains, stationarity tests, and classical decomposition.
//!
//! ## Modules
//! - [`forecast`] — [`Forecast`] / [`Forecaster`]
//! - [`baselines`] — naive, seasonal naive, drift
//! - [`ets`] — ETS state-space smoothers
//! - [`arima`] — ARIMA(p,d,q)
//! - [`prophet`] — Prophet-style OLS additive model (not Stan Prophet)
//! - [`markov`] — discrete-time Markov chains
//! - [`stationarity`] — ADF, KPSS
//! - [`decomposition`] — classical seasonal decomposition
//! - [`acf`] — ACF / PACF
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for overview and examples.
//!
//! ## Example
//! ```
//! use statscore_timeseries::baselines::drift;
//! use statscore_timeseries::forecast::Forecaster;
//! let f = drift(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();
//! assert!((f.point[0] - 5.0).abs() < 1e-12);
//! ```
//!
//! [`statscore-common`]: https://docs.rs/statscore-common

#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod acf;
pub mod arima;
pub mod baselines;
pub mod decomposition;
pub mod ets;
pub mod forecast;
pub mod markov;
pub mod prophet;
pub mod stationarity;
pub mod util;

pub use arima::ArimaModel;
pub use baselines::{DriftModel, NaiveModel, SeasonalNaiveModel, drift, naive, seasonal_naive};
pub use decomposition::{DecomposeModel, Decomposition, classical_decompose};
pub use ets::{EtsFit, EtsModel};
pub use forecast::{Forecast, Forecaster};
pub use markov::MarkovChain;
pub use prophet::{ProphetStyleModel, ProphetStyleSpec};
pub use stationarity::{AdfResult, KpssKind, KpssResult, adf_test, kpss_test};
