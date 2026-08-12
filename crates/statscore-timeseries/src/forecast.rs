//! Shared forecast output and [`Forecaster`] trait.

use statscore_common::Result;

/// Point forecast (and optional in-sample fit) from a time-series model.
#[derive(Debug, Clone, PartialEq)]
pub struct Forecast {
    /// `h`-step ahead point forecasts.
    pub point: Vec<f64>,
    /// In-sample fitted values (same length as the training series), if available.
    pub fitted: Option<Vec<f64>>,
    /// In-sample residuals `y − fitted`, if available.
    pub residuals: Option<Vec<f64>>,
}

impl Forecast {
    /// Build a forecast that only has point predictions.
    #[must_use]
    pub fn points(point: Vec<f64>) -> Self {
        Self {
            point,
            fitted: None,
            residuals: None,
        }
    }

    /// Build a forecast with fitted values and residuals.
    #[must_use]
    pub fn with_fit(point: Vec<f64>, fitted: Vec<f64>, residuals: Vec<f64>) -> Self {
        Self {
            point,
            fitted: Some(fitted),
            residuals: Some(residuals),
        }
    }
}

/// Models that can produce an `h`-step ahead forecast from a fitted state.
pub trait Forecaster {
    /// Forecast `h` steps ahead.
    ///
    /// # Errors
    /// Returns a [`StatsError`](statscore_common::StatsError) if `h == 0` or the
    /// model state is invalid.
    fn forecast(&self, h: usize) -> Result<Forecast>;
}
