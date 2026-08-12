//! Classical seasonal decomposition.

use statscore_common::{Result, StatsError, require_positive};

use crate::util::require_series;

/// Additive vs multiplicative classical decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecomposeModel {
    /// `y = trend + seasonal + residual`.
    Additive,
    /// `y = trend * seasonal * residual`.
    Multiplicative,
}

/// Result of classical decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct Decomposition {
    /// Estimated trend (NaN where the moving average is undefined at edges).
    pub trend: Vec<f64>,
    /// Seasonal component (repeated seasonal indices).
    pub seasonal: Vec<f64>,
    /// Residual / irregular component.
    pub residual: Vec<f64>,
    /// Seasonal period.
    pub period: usize,
    /// Model type used.
    pub model: DecomposeModel,
}

/// Classical seasonal decomposition via centered moving average.
///
/// # Errors
/// Requires `x.len() >= 2 * period`.
///
/// # Example
/// ```
/// use statscore_timeseries::decomposition::{classical_decompose, DecomposeModel};
/// let x: Vec<f64> = (0..48).map(|i| (i as f64) + 2.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()).collect();
/// let d = classical_decompose(&x, 12, DecomposeModel::Additive).unwrap();
/// assert_eq!(d.seasonal.len(), x.len());
/// ```
pub fn classical_decompose(
    x: &[f64],
    period: usize,
    model: DecomposeModel,
) -> Result<Decomposition> {
    require_positive(period as f64, "period")?;
    require_series(x, 2 * period, "classical_decompose")?;
    let n = x.len();

    // Centered moving average of window `period` (even → 2x period convolution light approx)
    let mut trend = vec![f64::NAN; n];
    if period % 2 == 1 {
        let half = period / 2;
        for t in half..n - half {
            let s: f64 = x[t - half..=t + half].iter().sum();
            trend[t] = s / period as f64;
        }
    } else {
        let half = period / 2;
        for t in half..n - half {
            // 2x(m/2) MA then average adjacent for centering
            let left: f64 = x[t - half..t + half].iter().sum::<f64>() / period as f64;
            let right: f64 = x[t - half + 1..=t + half].iter().sum::<f64>() / period as f64;
            trend[t] = 0.5 * (left + right);
        }
    }

    let mut raw_season = vec![0.0; period];
    let mut counts = vec![0usize; period];
    for t in 0..n {
        if trend[t].is_nan() {
            continue;
        }
        let idx = t % period;
        match model {
            DecomposeModel::Additive => {
                raw_season[idx] += x[t] - trend[t];
                counts[idx] += 1;
            }
            DecomposeModel::Multiplicative => {
                if trend[t].abs() < 1e-15 {
                    return Err(StatsError::domain(
                        "multiplicative decompose: near-zero trend",
                    ));
                }
                raw_season[idx] += x[t] / trend[t];
                counts[idx] += 1;
            }
        }
    }
    for i in 0..period {
        if counts[i] == 0 {
            return Err(StatsError::domain(
                "classical_decompose: could not estimate a seasonal index",
            ));
        }
        raw_season[i] /= counts[i] as f64;
    }
    // Normalize seasonal indices
    match model {
        DecomposeModel::Additive => {
            let mean_s = raw_season.iter().sum::<f64>() / period as f64;
            for s in &mut raw_season {
                *s -= mean_s;
            }
        }
        DecomposeModel::Multiplicative => {
            let mean_s = raw_season.iter().sum::<f64>() / period as f64;
            if mean_s.abs() < 1e-15 {
                return Err(StatsError::domain("seasonal indices sum to zero"));
            }
            for s in &mut raw_season {
                *s /= mean_s;
            }
        }
    }

    let seasonal: Vec<f64> = (0..n).map(|t| raw_season[t % period]).collect();
    let residual: Vec<f64> = (0..n)
        .map(|t| {
            if trend[t].is_nan() {
                f64::NAN
            } else {
                match model {
                    DecomposeModel::Additive => x[t] - trend[t] - seasonal[t],
                    DecomposeModel::Multiplicative => {
                        if (trend[t] * seasonal[t]).abs() < 1e-15 {
                            f64::NAN
                        } else {
                            x[t] / (trend[t] * seasonal[t])
                        }
                    }
                }
            }
        })
        .collect();

    Ok(Decomposition {
        trend,
        seasonal,
        residual,
        period,
        model,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn additive_runs() {
        let x: Vec<f64> = (0..36)
            .map(|i| i as f64 + (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let d = classical_decompose(&x, 12, DecomposeModel::Additive).unwrap();
        assert_eq!(d.trend.len(), 36);
        assert!(d.seasonal.iter().any(|s| s.abs() > 0.01));
    }
}
