//! AR / ARIMA models (pragmatic Yule–Walker + innovations MA).

use statscore_common::{Result, StatsError};
use statscore_linalg::matrix::{from_row_slice, square_from_row_slice, vector_from_slice};
use statscore_linalg::solve::{solve_least_squares, solve_linear_system};

use crate::acf::acf;
use crate::forecast::{Forecast, Forecaster};
use crate::util::{diff_n, mean, require_horizon, require_series, undiff, variance};

/// Fitted ARIMA(p, d, q) model.
#[derive(Debug, Clone)]
pub struct ArimaModel {
    /// AR order.
    pub p: usize,
    /// Differencing order.
    pub d: usize,
    /// MA order.
    pub q: usize,
    /// AR coefficients φ₁…φₚ.
    pub ar: Vec<f64>,
    /// MA coefficients θ₁…θ_q.
    pub ma: Vec<f64>,
    /// Intercept on the differenced series (mean).
    pub intercept: f64,
    /// Innovation variance.
    pub sigma2: f64,
    /// Last `p` differenced observations (for forecasting).
    history: Vec<f64>,
    /// Last `q` residuals.
    resid_hist: Vec<f64>,
    /// Levels needed to undifference forecasts (last `d` levels).
    levels: Vec<f64>,
    fitted: Vec<f64>,
    residuals: Vec<f64>,
    nobs: usize,
}

impl ArimaModel {
    /// Fit ARIMA(p,d,q) via Yule–Walker AR and innovations MA on residuals.
    ///
    /// # Errors
    /// Returns an error if orders are inconsistent with series length.
    ///
    /// # Example
    /// ```
    /// use statscore_timeseries::arima::ArimaModel;
    /// use statscore_timeseries::forecast::Forecaster;
    /// let mut x = vec![0.0];
    /// for i in 1..80 {
    ///     x.push(0.5 * x[i - 1] + 0.1 * (i as f64).sin());
    /// }
    /// let m = ArimaModel::fit(&x, 1, 0, 0).unwrap();
    /// assert_eq!(m.forecast(3).unwrap().point.len(), 3);
    /// ```
    pub fn fit(x: &[f64], p: usize, d: usize, q: usize) -> Result<Self> {
        require_series(x, p.max(q) + d + 5, "arima")?;
        let z = diff_n(x, d)?;
        let n = z.len();
        let intercept = mean(&z)?;

        // demean for AR estimation
        let zc: Vec<f64> = z.iter().map(|v| v - intercept).collect();

        let ar = if p == 0 { vec![] } else { yule_walker(&zc, p)? };

        // AR residuals
        let mut ar_resid = vec![0.0; n];
        for t in 0..n {
            let mut pred = intercept;
            for i in 0..p {
                if t > i {
                    pred += ar[i] * z[t - 1 - i];
                }
            }
            ar_resid[t] = z[t] - pred;
        }

        let ma = if q == 0 {
            vec![]
        } else {
            innovations_ma(&ar_resid, q)?
        };

        // final residuals with MA
        let mut residuals_z = vec![0.0; n];
        let mut e_hist = vec![0.0; q];
        for t in 0..n {
            let mut pred = intercept;
            for i in 0..p {
                if t > i {
                    pred += ar[i] * z[t - 1 - i];
                }
            }
            for i in 0..q {
                pred += ma[i] * e_hist[i];
            }
            let e = z[t] - pred;
            residuals_z[t] = e;
            if q > 0 {
                e_hist.rotate_right(1);
                e_hist[0] = e;
            }
        }

        let sigma2 = variance(&residuals_z)?;
        let history = if p == 0 {
            vec![]
        } else {
            z[n.saturating_sub(p)..].to_vec()
        };
        let resid_hist = if q == 0 {
            vec![]
        } else {
            residuals_z[n.saturating_sub(q)..].to_vec()
        };
        let levels = x[x.len().saturating_sub(d.max(1))..].to_vec();

        // map fitted differenced back to levels approximately for d=0; for d>0
        // report NA-style by using undiff of predicted z
        let (fitted, residuals) = if d == 0 {
            let fitted: Vec<f64> = z.iter().zip(&residuals_z).map(|(zi, e)| zi - e).collect();
            let residuals: Vec<f64> = x.iter().zip(&fitted).map(|(y, f)| y - f).collect();
            (fitted, residuals)
        } else if d == 1 {
            let pred_z: Vec<f64> = z.iter().zip(&residuals_z).map(|(zi, e)| zi - e).collect();
            let fitted = undiff(x[0], &pred_z);
            let residuals: Vec<f64> = x.iter().zip(&fitted).map(|(y, f)| y - f).collect();
            (fitted, residuals)
        } else {
            (x.to_vec(), vec![0.0; x.len()])
        };

        Ok(Self {
            p,
            d,
            q,
            ar,
            ma,
            intercept,
            sigma2,
            history,
            resid_hist,
            levels,
            fitted,
            residuals,
            nobs: n,
        })
    }

    /// Akaike information criterion (Gaussian).
    #[must_use]
    pub fn aic(&self) -> f64 {
        let k = (self.p + self.q + 1) as f64; // + intercept
        let n = self.nobs as f64;
        if self.sigma2 <= 0.0 {
            return f64::INFINITY;
        }
        n * self.sigma2.ln() + 2.0 * k
    }

    /// Bayesian information criterion.
    #[must_use]
    pub fn bic(&self) -> f64 {
        let k = (self.p + self.q + 1) as f64;
        let n = self.nobs as f64;
        if self.sigma2 <= 0.0 {
            return f64::INFINITY;
        }
        n * self.sigma2.ln() + k * n.ln()
    }
}

impl Forecaster for ArimaModel {
    fn forecast(&self, h: usize) -> Result<Forecast> {
        require_horizon(h)?;
        let mut hist = self.history.clone();
        let mut e_hist = self.resid_hist.clone();
        let mut z_fc = Vec::with_capacity(h);
        for _ in 0..h {
            let mut pred = self.intercept;
            for i in 0..self.p {
                if i < hist.len() {
                    pred += self.ar[i] * hist[hist.len() - 1 - i];
                }
            }
            for i in 0..self.q {
                if i < e_hist.len() {
                    pred += self.ma[i] * e_hist[e_hist.len() - 1 - i];
                }
            }
            z_fc.push(pred);
            if self.p > 0 {
                hist.push(pred);
                if hist.len() > self.p {
                    hist.remove(0);
                }
            }
            if self.q > 0 {
                e_hist.push(0.0); // future shocks = 0
                if e_hist.len() > self.q {
                    e_hist.remove(0);
                }
            }
        }

        let point = match self.d {
            0 => z_fc,
            1 => {
                let mut lvl = *self.levels.last().unwrap_or(&0.0);
                z_fc.iter()
                    .map(|dz| {
                        lvl += dz;
                        lvl
                    })
                    .collect()
            }
            _ => {
                // iterative undiff using last d levels
                let mut levels = self.levels.clone();
                let mut out = Vec::with_capacity(h);
                for &dz in &z_fc {
                    // for d>1 this is approximate: treat as d=1 on last level
                    let last = *levels.last().unwrap_or(&0.0);
                    let next = last + dz;
                    levels.push(next);
                    out.push(next);
                }
                out
            }
        };

        Ok(Forecast::with_fit(
            point,
            self.fitted.clone(),
            self.residuals.clone(),
        ))
    }
}

fn yule_walker(zc: &[f64], p: usize) -> Result<Vec<f64>> {
    let r = acf(zc, p)?;
    // Toeplitz R φ = r
    let mut mat = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            let lag = i.abs_diff(j);
            mat[i * p + j] = if lag == 0 { 1.0 } else { r[lag - 1] };
        }
    }
    let a = square_from_row_slice(p, &mat)?;
    let b = vector_from_slice(&r[..p]);
    let phi = solve_linear_system(&a, &b)?;
    Ok((0..p).map(|i| phi.get(i)).collect())
}

fn innovations_ma(resid: &[f64], q: usize) -> Result<Vec<f64>> {
    // Approximate θ by regressing e_t on e_{t-1}…e_{t-q}
    let n = resid.len();
    if n <= q + 1 {
        return Err(StatsError::insufficient_data(q + 2, n));
    }
    let nobs = n - q;
    let mut design = Vec::with_capacity(nobs * q);
    let mut y = Vec::with_capacity(nobs);
    for t in q..n {
        y.push(resid[t]);
        for j in 1..=q {
            design.push(resid[t - j]);
        }
    }
    let a = from_row_slice(nobs, q, &design)?;
    let b = vector_from_slice(&y);
    let theta = solve_least_squares(&a, &b)?;
    Ok((0..q).map(|i| theta.get(i)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ar1_recovers_phi() {
        let phi_true = 0.5;
        let mut x = vec![0.0];
        for i in 1..200 {
            x.push(phi_true * x[i - 1] + 0.01 * ((i % 7) as f64 - 3.0));
        }
        let m = ArimaModel::fit(&x, 1, 0, 0).unwrap();
        assert_relative_eq!(m.ar[0], phi_true, epsilon = 0.15);
    }

    #[test]
    fn arima_010_like_naive() {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let m = ArimaModel::fit(&x, 0, 1, 0).unwrap();
        let f = m.forecast(1).unwrap();
        // differenced mean ~1, so next ≈ last+1
        assert_relative_eq!(f.point[0], 30.0, epsilon = 0.5);
    }
}
