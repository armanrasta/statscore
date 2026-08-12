//! Forecast baselines: naive, seasonal naive, and drift.

use statscore_common::{Result, require_positive};

use crate::forecast::{Forecast, Forecaster};
use crate::util::{require_horizon, require_series};

/// Fitted naive (last-value) forecaster.
#[derive(Debug, Clone)]
pub struct NaiveModel {
    last: f64,
    fitted: Vec<f64>,
    residuals: Vec<f64>,
}

impl NaiveModel {
    /// Fit a naive model: forecast equals the last observation.
    ///
    /// # Errors
    /// Returns an error if the series is empty or non-finite.
    ///
    /// # Example
    /// ```
    /// use statscore_timeseries::baselines::NaiveModel;
    /// use statscore_timeseries::forecast::Forecaster;
    /// let m = NaiveModel::fit(&[1.0, 2.0, 5.0]).unwrap();
    /// assert_eq!(m.forecast(2).unwrap().point, vec![5.0, 5.0]);
    /// ```
    pub fn fit(x: &[f64]) -> Result<Self> {
        require_series(x, 1, "naive")?;
        let last = *x.last().unwrap();
        // Standard naive fitted: ŷ_t = y_{t-1} for t≥1, ŷ_0 = y_0
        let fitted: Vec<f64> = (0..x.len())
            .map(|t| if t == 0 { x[0] } else { x[t - 1] })
            .collect();
        let residuals: Vec<f64> = x.iter().zip(&fitted).map(|(y, f)| y - f).collect();
        Ok(Self {
            last,
            fitted,
            residuals,
        })
    }
}

impl Forecaster for NaiveModel {
    fn forecast(&self, h: usize) -> Result<Forecast> {
        require_horizon(h)?;
        Ok(Forecast::with_fit(
            vec![self.last; h],
            self.fitted.clone(),
            self.residuals.clone(),
        ))
    }
}

/// Seasonal naive: repeat the last full season.
#[derive(Debug, Clone)]
pub struct SeasonalNaiveModel {
    period: usize,
    season: Vec<f64>,
    fitted: Vec<f64>,
    residuals: Vec<f64>,
}

impl SeasonalNaiveModel {
    /// Fit seasonal naive with seasonal period `period`.
    ///
    /// # Errors
    /// Requires `x.len() >= period` and `period >= 1`.
    pub fn fit(x: &[f64], period: usize) -> Result<Self> {
        require_positive(period as f64, "period")?;
        require_series(x, period, "seasonal_naive")?;
        let season = x[x.len() - period..].to_vec();
        let fitted: Vec<f64> = (0..x.len())
            .map(|t| {
                if t < period {
                    x[t]
                } else {
                    x[t - period]
                }
            })
            .collect();
        let residuals: Vec<f64> = x.iter().zip(&fitted).map(|(y, f)| y - f).collect();
        Ok(Self {
            period,
            season,
            fitted,
            residuals,
        })
    }
}

impl Forecaster for SeasonalNaiveModel {
    fn forecast(&self, h: usize) -> Result<Forecast> {
        require_horizon(h)?;
        let point: Vec<f64> = (0..h)
            .map(|i| self.season[i % self.period])
            .collect();
        Ok(Forecast::with_fit(
            point,
            self.fitted.clone(),
            self.residuals.clone(),
        ))
    }
}

/// Drift model: straight line from first to last observation.
#[derive(Debug, Clone)]
pub struct DriftModel {
    last: f64,
    slope: f64,
    fitted: Vec<f64>,
    residuals: Vec<f64>,
}

impl DriftModel {
    /// Fit a drift model.
    ///
    /// # Errors
    /// Requires at least two observations.
    ///
    /// # Example
    /// ```
    /// use statscore_timeseries::baselines::DriftModel;
    /// use statscore_timeseries::forecast::Forecaster;
    /// let m = DriftModel::fit(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let f = m.forecast(2).unwrap();
    /// assert!((f.point[0] - 5.0).abs() < 1e-12);
    /// ```
    pub fn fit(x: &[f64]) -> Result<Self> {
        require_series(x, 2, "drift")?;
        let n = x.len();
        let slope = (x[n - 1] - x[0]) / (n as f64 - 1.0);
        let fitted: Vec<f64> = (0..n)
            .map(|t| x[0] + slope * t as f64)
            .collect();
        let residuals: Vec<f64> = x.iter().zip(&fitted).map(|(y, f)| y - f).collect();
        Ok(Self {
            last: x[n - 1],
            slope,
            fitted,
            residuals,
        })
    }
}

impl Forecaster for DriftModel {
    fn forecast(&self, h: usize) -> Result<Forecast> {
        require_horizon(h)?;
        let point: Vec<f64> = (1..=h)
            .map(|i| self.last + self.slope * i as f64)
            .collect();
        Ok(Forecast::with_fit(
            point,
            self.fitted.clone(),
            self.residuals.clone(),
        ))
    }
}

/// Convenience: naive forecast without keeping the model.
pub fn naive(x: &[f64], h: usize) -> Result<Forecast> {
    NaiveModel::fit(x)?.forecast(h)
}

/// Convenience: seasonal naive forecast.
pub fn seasonal_naive(x: &[f64], period: usize, h: usize) -> Result<Forecast> {
    SeasonalNaiveModel::fit(x, period)?.forecast(h)
}

/// Convenience: drift forecast.
pub fn drift(x: &[f64], h: usize) -> Result<Forecast> {
    DriftModel::fit(x)?.forecast(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn naive_repeats_last() {
        let f = naive(&[1.0, 2.0, 9.0], 3).unwrap();
        assert_eq!(f.point, vec![9.0, 9.0, 9.0]);
    }

    #[test]
    fn seasonal_naive_period() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f = seasonal_naive(&x, 3, 4).unwrap();
        assert_eq!(f.point, vec![4.0, 5.0, 6.0, 4.0]);
    }

    #[test]
    fn drift_linear() {
        let f = drift(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();
        assert_relative_eq!(f.point[0], 5.0, epsilon = 1e-12);
        assert_relative_eq!(f.point[1], 6.0, epsilon = 1e-12);
    }

    #[test]
    fn horizon_zero_errors() {
        assert!(naive(&[1.0], 0).is_err());
    }
}
