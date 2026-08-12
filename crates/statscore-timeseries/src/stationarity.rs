//! Stationarity tests: Augmented Dickey–Fuller and KPSS.

use statscore_common::{Result, StatsError};
use statscore_linalg::matrix::{from_row_slice, vector_from_slice};
use statscore_linalg::solve::solve_least_squares;

use crate::util::{diff, require_series};

/// Result of an Augmented Dickey–Fuller test (no constant/trend for simplicity:
/// regression of `Δy` on `y_{t−1}` and lag differences).
#[derive(Debug, Clone, PartialEq)]
pub struct AdfResult {
    /// ADF test statistic (t-ratio on the lagged level coefficient).
    pub statistic: f64,
    /// Number of observations used in the regression.
    pub nobs: usize,
    /// Number of lagged differences included.
    pub lags: usize,
}

/// Augmented Dickey–Fuller unit-root test.
///
/// Fits `Δy_t = γ y_{t−1} + Σ φ_i Δy_{t−i} + ε_t` and returns the t-statistic for `γ`.
/// More negative values suggest stationarity (reject unit root).
///
/// # Errors
/// Returns an error if the series is too short for the chosen lag order.
///
/// # Example
/// ```
/// use statscore_timeseries::stationarity::adf_test;
/// let x: Vec<f64> = (0..80).map(|i| ((i * 13) % 7) as f64).collect();
/// let r = adf_test(&x, Some(1)).unwrap();
/// assert!(r.nobs > 0);
/// ```
pub fn adf_test(x: &[f64], max_lag: Option<usize>) -> Result<AdfResult> {
    require_series(x, 10, "adf_test")?;
    let lags = max_lag.unwrap_or_else(|| {
        // Schwert rule of thumb
        ((x.len() as f64).powf(1.0 / 3.0) * 12.0 / 100.0).floor() as usize
    });
    let dx = diff(x)?;
    let n = x.len();
    // Effective sample: from index (lags+1) .. n-1 in levels → dx index lags .. n-2
    let start = lags;
    let nobs = n - 1 - start;
    if nobs < lags + 2 {
        return Err(StatsError::insufficient_data(lags + 2, nobs));
    }

    let ncols = 1 + lags;
    let mut design = Vec::with_capacity(nobs * ncols);
    let mut y = Vec::with_capacity(nobs);
    for t in start..(n - 1) {
        // Δy_t at index t in dx (dx[t] = x[t+1]-x[t])
        y.push(dx[t]);
        design.push(x[t]); // y_{t}
        for lag in 1..=lags {
            design.push(dx[t - lag]);
        }
    }

    let a = from_row_slice(nobs, ncols, &design)?;
    let b = vector_from_slice(&y);
    let beta = solve_least_squares(&a, &b)?;
    let gamma = beta.get(0);

    // Residual SE and (X'X)^{-1}_{00} for t-stat
    let mut sse = 0.0;
    for i in 0..nobs {
        let mut pred = 0.0;
        for j in 0..ncols {
            pred += beta.get(j) * design[i * ncols + j];
        }
        sse += (y[i] - pred).powi(2);
    }
    let df = (nobs as i64 - ncols as i64).max(1) as f64;
    let sigma2 = sse / df;

    // Build X'X and invert for var(γ)
    let mut xtx = vec![0.0; ncols * ncols];
    for i in 0..nobs {
        for r in 0..ncols {
            for c in 0..ncols {
                xtx[r * ncols + c] += design[i * ncols + r] * design[i * ncols + c];
            }
        }
    }
    let xtx_mat = statscore_linalg::matrix::square_from_row_slice(ncols, &xtx)?;
    let e0 = vector_from_slice(&{
        let mut e = vec![0.0; ncols];
        e[0] = 1.0;
        e
    });
    let inv_col = statscore_linalg::solve::solve_linear_system(&xtx_mat, &e0)?;
    let var_gamma = sigma2 * inv_col.get(0);
    let se = var_gamma.max(0.0).sqrt();
    let statistic = if se < 1e-15 { f64::NAN } else { gamma / se };

    Ok(AdfResult {
        statistic,
        nobs,
        lags,
    })
}

/// KPSS regression type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssKind {
    /// Level stationarity (constant only).
    Level,
    /// Trend stationarity (constant + linear trend).
    Trend,
}

/// Result of a KPSS test.
#[derive(Debug, Clone, PartialEq)]
pub struct KpssResult {
    /// KPSS statistic (larger ⇒ more evidence against stationarity).
    pub statistic: f64,
    /// Number of observations.
    pub nobs: usize,
}

/// KPSS stationarity test (null: series is stationary).
///
/// # Errors
/// Returns an error if the series is too short.
pub fn kpss_test(x: &[f64], kind: KpssKind) -> Result<KpssResult> {
    require_series(x, 5, "kpss_test")?;
    let n = x.len();
    let residuals: Vec<f64> = match kind {
        KpssKind::Level => {
            let m = x.iter().sum::<f64>() / n as f64;
            x.iter().map(|v| v - m).collect()
        }
        KpssKind::Trend => {
            // OLS: y = a + b t
            let t_mean = (n as f64 - 1.0) / 2.0;
            let y_mean = x.iter().sum::<f64>() / n as f64;
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            for (i, &y) in x.iter().enumerate() {
                let dt = i as f64 - t_mean;
                sxx += dt * dt;
                sxy += dt * (y - y_mean);
            }
            let b = if sxx == 0.0 { 0.0 } else { sxy / sxx };
            let a = y_mean - b * t_mean;
            x.iter()
                .enumerate()
                .map(|(i, &y)| y - a - b * i as f64)
                .collect()
        }
    };

    let mut partial = 0.0;
    let mut eta = 0.0;
    for &e in &residuals {
        partial += e;
        eta += partial * partial;
    }
    let sigma2 = residuals.iter().map(|e| e * e).sum::<f64>() / n as f64;
    if sigma2 <= 0.0 {
        return Ok(KpssResult {
            statistic: 0.0,
            nobs: n,
        });
    }
    let statistic = eta / (n as f64 * n as f64 * sigma2);
    Ok(KpssResult { statistic, nobs: n })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adf_on_white_noise_finite() {
        let x: Vec<f64> = (0..60).map(|i| ((i * 13) % 7) as f64).collect();
        let r = adf_test(&x, Some(1)).unwrap();
        assert!(r.statistic.is_finite());
    }

    #[test]
    fn kpss_level_runs() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64).sin()).collect();
        let r = kpss_test(&x, KpssKind::Level).unwrap();
        assert!(r.statistic >= 0.0);
    }
}
