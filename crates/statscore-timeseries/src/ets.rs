//! ETS (Error-Trend-Seasonal) exponential smoothing.

use statscore_common::{Result, StatsError, require_positive};

use crate::forecast::{Forecast, Forecaster};
use crate::util::{require_horizon, require_series};

/// ETS model class (additive/multiplicative error × trend × season).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EtsModel {
    /// Simple exponential smoothing (ANN).
    Ann,
    /// Holt linear (AAN).
    Aan,
    /// Additive Holt–Winters (AAA).
    Aaa,
    /// Multiplicative error, no trend/season (MNN) — treated as SES on levels.
    Mnn,
}

/// Fitted ETS model.
#[derive(Debug, Clone)]
pub struct EtsFit {
    model: EtsModel,
    alpha: f64,
    beta: f64,
    gamma: f64,
    period: usize,
    level: f64,
    trend: f64,
    season: Vec<f64>,
    fitted: Vec<f64>,
    residuals: Vec<f64>,
}

impl EtsFit {
    /// Fit an ETS model with default smoothing parameters.
    ///
    /// Defaults: `α=0.3`, `β=0.1`, `γ=0.1`. For seasonal models `period` is required.
    ///
    /// # Errors
    /// Returns an error if the series is too short or `period` is invalid.
    ///
    /// # Example
    /// ```
    /// use statscore_timeseries::ets::{EtsFit, EtsModel};
    /// use statscore_timeseries::forecast::Forecaster;
    /// let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    /// let m = EtsFit::fit(&x, EtsModel::Aan, None).unwrap();
    /// assert_eq!(m.forecast(3).unwrap().point.len(), 3);
    /// ```
    pub fn fit(x: &[f64], model: EtsModel, period: Option<usize>) -> Result<Self> {
        match model {
            EtsModel::Ann | EtsModel::Mnn => fit_ses(x, 0.3, model),
            EtsModel::Aan => fit_holt(x, 0.3, 0.1),
            EtsModel::Aaa => {
                let p = period
                    .ok_or_else(|| StatsError::domain("ETS AAA requires a seasonal period"))?;
                fit_holt_winters_additive(x, p, 0.3, 0.1, 0.1)
            }
        }
    }

    /// Model class.
    #[must_use]
    pub fn model(&self) -> EtsModel {
        self.model
    }

    /// Level smoothing parameter α.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Trend smoothing parameter β.
    #[must_use]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Seasonal smoothing parameter γ.
    #[must_use]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

impl Forecaster for EtsFit {
    fn forecast(&self, h: usize) -> Result<Forecast> {
        require_horizon(h)?;
        let point = match self.model {
            EtsModel::Ann | EtsModel::Mnn => vec![self.level; h],
            EtsModel::Aan => (1..=h)
                .map(|i| self.level + self.trend * i as f64)
                .collect(),
            EtsModel::Aaa => (1..=h)
                .map(|i| {
                    let idx = (i - 1) % self.period;
                    self.level + self.trend * i as f64 + self.season[idx]
                })
                .collect(),
        };
        Ok(Forecast::with_fit(
            point,
            self.fitted.clone(),
            self.residuals.clone(),
        ))
    }
}

fn fit_ses(x: &[f64], alpha: f64, model: EtsModel) -> Result<EtsFit> {
    require_series(x, 2, "ets SES")?;
    require_positive(alpha, "alpha")?;
    let mut level = x[0];
    let mut fitted = Vec::with_capacity(x.len());
    fitted.push(level);
    for &y in &x[1..] {
        let f = level;
        fitted.push(f);
        level = alpha * y + (1.0 - alpha) * level;
    }
    let residuals: Vec<f64> = x.iter().zip(&fitted).map(|(y, f)| y - f).collect();
    Ok(EtsFit {
        model,
        alpha,
        beta: 0.0,
        gamma: 0.0,
        period: 0,
        level,
        trend: 0.0,
        season: vec![],
        fitted,
        residuals,
    })
}

fn fit_holt(x: &[f64], alpha: f64, beta: f64) -> Result<EtsFit> {
    require_series(x, 3, "ets Holt")?;
    let mut level = x[0];
    let mut trend = x[1] - x[0];
    let mut fitted = Vec::with_capacity(x.len());
    fitted.push(level);
    for (t, &y) in x.iter().enumerate().skip(1) {
        let f = level + trend;
        fitted.push(f);
        let new_level = alpha * y + (1.0 - alpha) * (level + trend);
        trend = beta * (new_level - level) + (1.0 - beta) * trend;
        level = new_level;
        let _ = t;
    }
    let residuals: Vec<f64> = x.iter().zip(&fitted).map(|(y, f)| y - f).collect();
    Ok(EtsFit {
        model: EtsModel::Aan,
        alpha,
        beta,
        gamma: 0.0,
        period: 0,
        level,
        trend,
        season: vec![],
        fitted,
        residuals,
    })
}

fn fit_holt_winters_additive(
    x: &[f64],
    period: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
) -> Result<EtsFit> {
    require_positive(period as f64, "period")?;
    require_series(x, 2 * period, "ets Holt-Winters")?;
    let mut season = vec![0.0; period];
    // init seasonal: demean first season
    let first_mean = x[..period].iter().sum::<f64>() / period as f64;
    for i in 0..period {
        season[i] = x[i] - first_mean;
    }
    let mut level = first_mean;
    let mut trend = {
        let second: f64 = x[period..2 * period].iter().sum::<f64>() / period as f64;
        (second - first_mean) / period as f64
    };

    let mut fitted = Vec::with_capacity(x.len());
    for t in 0..x.len() {
        let s = season[t % period];
        let f = level + trend + s;
        fitted.push(f);
        if t + 1 < x.len() || t >= period {
            let y = x[t];
            let new_level = alpha * (y - s) + (1.0 - alpha) * (level + trend);
            let new_trend = beta * (new_level - level) + (1.0 - beta) * trend;
            season[t % period] = gamma * (y - new_level) + (1.0 - gamma) * s;
            level = new_level;
            trend = new_trend;
        }
    }
    // rotate season so index 0 is next season factor
    let next_start = x.len() % period;
    let mut season_ord = Vec::with_capacity(period);
    for i in 0..period {
        season_ord.push(season[(next_start + i) % period]);
    }
    let residuals: Vec<f64> = x.iter().zip(&fitted).map(|(y, f)| y - f).collect();
    Ok(EtsFit {
        model: EtsModel::Aaa,
        alpha,
        beta,
        gamma,
        period,
        level,
        trend,
        season: season_ord,
        fitted,
        residuals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn holt_tracks_linear() {
        let x: Vec<f64> = (0..40).map(|i| 2.0 * i as f64).collect();
        let m = EtsFit::fit(&x, EtsModel::Aan, None).unwrap();
        let f = m.forecast(5).unwrap();
        // slope ~2
        assert_relative_eq!(f.point[0] - x[x.len() - 1], 2.0, epsilon = 0.5);
    }

    #[test]
    fn aaa_needs_period() {
        assert!(EtsFit::fit(&[1.0, 2.0, 3.0], EtsModel::Aaa, None).is_err());
    }
}
