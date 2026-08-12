//! Prophet-style additive regression (OLS, not Stan/Bayesian Prophet).

use std::f64::consts::PI;

use statscore_common::{Result, StatsError, require_min_len};
use statscore_linalg::matrix::{from_row_slice, vector_from_slice};
use statscore_linalg::solve::solve_least_squares;

use crate::forecast::Forecast;
use crate::util::require_series;

/// Specification for a Prophet-style model.
#[derive(Debug, Clone)]
pub struct ProphetStyleSpec {
    /// Number of uniformly spaced changepoints in `(t_min, t_max)`.
    pub n_changepoints: usize,
    /// Number of Fourier pairs for seasonality (order).
    pub fourier_order: usize,
    /// Seasonal period in the same units as `t` (e.g. 365.25 for yearly days).
    pub period: f64,
}

impl Default for ProphetStyleSpec {
    fn default() -> Self {
        Self {
            n_changepoints: 5,
            fourier_order: 3,
            period: 365.25,
        }
    }
}

/// Fitted Prophet-style additive model: piecewise linear trend + Fourier seasonality.
#[derive(Debug, Clone)]
pub struct ProphetStyleModel {
    /// Intercept.
    pub k0: f64,
    /// Base slope.
    pub m: f64,
    /// Slope offsets at each changepoint.
    pub deltas: Vec<f64>,
    /// Changepoint locations.
    pub changepoints: Vec<f64>,
    /// Fourier coefficients interleaved [a1,b1,a2,b2,…].
    pub fourier_coef: Vec<f64>,
    spec: ProphetStyleSpec,
    fitted: Vec<f64>,
    residuals: Vec<f64>,
}

impl ProphetStyleModel {
    /// Fit by ordinary least squares.
    ///
    /// `t` and `y` must have the same length. This is **not** Facebook Prophet
    /// (no MCMC / uncertainty intervals).
    ///
    /// # Errors
    /// Returns an error on length mismatch or insufficient data.
    ///
    /// # Example
    /// ```
    /// use statscore_timeseries::prophet::{ProphetStyleModel, ProphetStyleSpec};
    /// let t: Vec<f64> = (0..60).map(|i| i as f64).collect();
    /// let y: Vec<f64> = t.iter().map(|&ti| 1.0 + 0.1 * ti + (2.0 * std::f64::consts::PI * ti / 12.0).sin()).collect();
    /// let mut spec = ProphetStyleSpec::default();
    /// spec.period = 12.0;
    /// spec.n_changepoints = 3;
    /// let m = ProphetStyleModel::fit(&t, &y, &spec).unwrap();
    /// assert!(m.predict(&[60.0, 61.0]).unwrap().point.len() == 2);
    /// ```
    pub fn fit(t: &[f64], y: &[f64], spec: &ProphetStyleSpec) -> Result<Self> {
        require_series(y, 8, "prophet y")?;
        require_min_len(t, y.len())?;
        if t.len() != y.len() {
            return Err(StatsError::dim_mismatch(format!(
                "prophet: t and y length differ ({} vs {})",
                t.len(),
                y.len()
            )));
        }
        if spec.period <= 0.0 {
            return Err(StatsError::domain("prophet period must be positive"));
        }

        let n = y.len();
        let t_min = t.iter().cloned().fold(f64::INFINITY, f64::min);
        let t_max = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let span = (t_max - t_min).max(1e-9);

        let n_cp = spec.n_changepoints;
        let mut changepoints = Vec::with_capacity(n_cp);
        for i in 0..n_cp {
            let u = (i as f64 + 1.0) / (n_cp as f64 + 1.0);
            changepoints.push(t_min + u * span);
        }

        let n_four = spec.fourier_order;
        // columns: intercept, slope t, n_cp changepoint ramps, 2*n_four fourier
        let ncols = 2 + n_cp + 2 * n_four;
        let mut design = vec![0.0; n * ncols];
        for i in 0..n {
            let ti = t[i];
            let row = i * ncols;
            design[row] = 1.0;
            design[row + 1] = ti;
            for (c, &cp) in changepoints.iter().enumerate() {
                design[row + 2 + c] = (ti - cp).max(0.0);
            }
            for k in 1..=n_four {
                let angle = 2.0 * PI * k as f64 * ti / spec.period;
                let base = row + 2 + n_cp + 2 * (k - 1);
                design[base] = angle.cos();
                design[base + 1] = angle.sin();
            }
        }

        let a = from_row_slice(n, ncols, &design)?;
        let b = vector_from_slice(y);
        let beta = solve_least_squares(&a, &b)?;

        let k0 = beta.get(0);
        let m = beta.get(1);
        let deltas: Vec<f64> = (0..n_cp).map(|i| beta.get(2 + i)).collect();
        let fourier_coef: Vec<f64> = (0..2 * n_four)
            .map(|i| beta.get(2 + n_cp + i))
            .collect();

        let fitted: Vec<f64> = (0..n)
            .map(|i| predict_one(t[i], k0, m, &deltas, &changepoints, &fourier_coef, spec))
            .collect();
        let residuals: Vec<f64> = y.iter().zip(&fitted).map(|(yi, fi)| yi - fi).collect();

        Ok(Self {
            k0,
            m,
            deltas,
            changepoints,
            fourier_coef,
            spec: spec.clone(),
            fitted,
            residuals,
        })
    }

    /// Predict at future time stamps `t_future`.
    ///
    /// # Errors
    /// Returns an error if `t_future` is empty.
    pub fn predict(&self, t_future: &[f64]) -> Result<Forecast> {
        if t_future.is_empty() {
            return Err(StatsError::domain("prophet predict: empty t_future"));
        }
        let point: Vec<f64> = t_future
            .iter()
            .map(|&ti| {
                predict_one(
                    ti,
                    self.k0,
                    self.m,
                    &self.deltas,
                    &self.changepoints,
                    &self.fourier_coef,
                    &self.spec,
                )
            })
            .collect();
        Ok(Forecast::with_fit(
            point,
            self.fitted.clone(),
            self.residuals.clone(),
        ))
    }
}

fn predict_one(
    ti: f64,
    k0: f64,
    m: f64,
    deltas: &[f64],
    cps: &[f64],
    four: &[f64],
    spec: &ProphetStyleSpec,
) -> f64 {
    let mut y = k0 + m * ti;
    for (d, &cp) in deltas.iter().zip(cps) {
        y += d * (ti - cp).max(0.0);
    }
    for k in 1..=spec.fourier_order {
        let angle = 2.0 * PI * k as f64 * ti / spec.period;
        let base = 2 * (k - 1);
        if base + 1 < four.len() {
            y += four[base] * angle.cos() + four[base + 1] * angle.sin();
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn prophet_style_tracks_trend_season() {
        let t: Vec<f64> = (0..80).map(|i| i as f64).collect();
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| 2.0 + 0.05 * ti + (2.0 * PI * ti / 16.0).sin())
            .collect();
        let mut spec = ProphetStyleSpec::default();
        spec.period = 16.0;
        spec.n_changepoints = 2;
        spec.fourier_order = 2;
        let m = ProphetStyleModel::fit(&t, &y, &spec).unwrap();
        let mse: f64 = m
            .residuals
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            / y.len() as f64;
        assert!(mse < 0.1, "mse={mse}");
        let f = m.predict(&[80.0]).unwrap();
        assert_relative_eq!(f.point[0], 2.0 + 0.05 * 80.0, epsilon = 1.5);
    }
}
