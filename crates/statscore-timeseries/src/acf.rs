//! Autocorrelation and partial autocorrelation.

use statscore_common::Result;

use crate::util::{mean, require_series, variance};

/// Sample autocorrelation function for lags `1..=max_lag`.
///
/// Returns a vector of length `max_lag` (lag 1 at index 0).
///
/// # Errors
/// Returns an error if the series is too short or non-finite.
///
/// # Example
/// ```
/// use statscore_timeseries::acf::acf;
/// let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
/// let rho = acf(&x, 5).unwrap();
/// assert_eq!(rho.len(), 5);
/// ```
pub fn acf(x: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    require_series(x, max_lag + 2, "acf")?;
    let m = mean(x)?;
    let var = variance(x)?;
    if var == 0.0 {
        return Ok(vec![0.0; max_lag]);
    }
    let n = x.len();
    let mut out = Vec::with_capacity(max_lag);
    for lag in 1..=max_lag {
        let mut cov = 0.0;
        for t in lag..n {
            cov += (x[t] - m) * (x[t - lag] - m);
        }
        out.push(cov / (n as f64 * var));
    }
    Ok(out)
}

/// Partial autocorrelation via Durbin–Levinson for lags `1..=max_lag`.
///
/// # Errors
/// Returns an error if ACF estimation fails.
pub fn pacf(x: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    let r = acf(x, max_lag)?;
    // Prefixed with r0 = 1
    let mut r_full = vec![1.0];
    r_full.extend_from_slice(&r);

    let mut phi = vec![0.0; max_lag];
    let mut phi_prev = vec![0.0; max_lag];

    for k in 1..=max_lag {
        let mut num = r_full[k];
        for j in 1..k {
            num -= phi_prev[j - 1] * r_full[k - j];
        }
        let mut den = 1.0;
        for j in 1..k {
            den -= phi_prev[j - 1] * r_full[j];
        }
        let pk = if den.abs() < 1e-15 { 0.0 } else { num / den };
        phi[k - 1] = pk;
        let mut phi_new = vec![0.0; k];
        for j in 1..k {
            phi_new[j - 1] = phi_prev[j - 1] - pk * phi_prev[k - j - 1];
        }
        phi_new[k - 1] = pk;
        for j in 0..k {
            phi_prev[j] = phi_new[j];
        }
    }
    Ok(phi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acf_white_noise_small() {
        let x: Vec<f64> = (0..100).map(|i| ((i * 17) % 10) as f64 - 4.5).collect();
        let rho = acf(&x, 3).unwrap();
        assert_eq!(rho.len(), 3);
        assert!(rho.iter().all(|v| v.abs() < 1.0));
    }
}
