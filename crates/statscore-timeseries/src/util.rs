//! Series validation and differencing helpers.

use statscore_common::{Result, StatsError, require_finite, require_min_len};

/// Validate a finite series of at least `min_len` observations.
///
/// # Errors
/// Returns [`StatsError::InsufficientData`] or [`StatsError::Domain`] on bad input.
pub fn require_series(x: &[f64], min_len: usize, ctx: &str) -> Result<()> {
    require_min_len(x, min_len)?;
    require_finite(x, ctx)?;
    Ok(())
}

/// First-order difference: `y[t] = x[t] − x[t−1]`.
///
/// # Errors
/// Returns [`StatsError::InsufficientData`] if `x.len() < 2`.
pub fn diff(x: &[f64]) -> Result<Vec<f64>> {
    require_min_len(x, 2)?;
    Ok(x.windows(2).map(|w| w[1] - w[0]).collect())
}

/// Apply first-order differencing `d` times.
///
/// # Errors
/// Returns an error if the series is too short for `d` differences.
pub fn diff_n(x: &[f64], d: usize) -> Result<Vec<f64>> {
    if d == 0 {
        return Ok(x.to_vec());
    }
    require_min_len(x, d + 1)?;
    let mut cur = x.to_vec();
    for _ in 0..d {
        cur = diff(&cur)?;
    }
    Ok(cur)
}

/// Invert one difference given the first level `x0` and differences `dx`.
#[must_use]
pub fn undiff(x0: f64, dx: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(dx.len() + 1);
    out.push(x0);
    for &d in dx {
        let prev = *out.last().unwrap();
        out.push(prev + d);
    }
    out
}

/// Sample mean of a non-empty finite slice.
///
/// # Errors
/// Returns [`StatsError::InsufficientData`] if empty.
pub fn mean(x: &[f64]) -> Result<f64> {
    require_min_len(x, 1)?;
    Ok(x.iter().sum::<f64>() / x.len() as f64)
}

/// Population variance (`/ n`).
///
/// # Errors
/// Returns [`StatsError::InsufficientData`] if empty.
pub fn variance(x: &[f64]) -> Result<f64> {
    let m = mean(x)?;
    let n = x.len() as f64;
    Ok(x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n)
}

/// Require `h ≥ 1`.
pub(crate) fn require_horizon(h: usize) -> Result<()> {
    if h == 0 {
        return Err(StatsError::domain("forecast horizon h must be >= 1"));
    }
    Ok(())
}
