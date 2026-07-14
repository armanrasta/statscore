//! Fuzzy statistics over triangular fuzzy numbers.
//!
//! The mean is computed vertex-wise (Zadeh's extension principle applied to the
//! triangular family), while variance and correlation defuzzify to crisp
//! summaries. On crisp data (degenerate fuzzy numbers) these reduce to the
//! classical population statistics.

use statscore_common::{Result, StatsError};

use crate::sets::TriangularFuzzyNumber;
use crate::traits::FuzzyNumber;

/// Fuzzy mean: the triangular number whose vertices are the means of the input
/// vertices.
///
/// # Errors
/// Returns [`StatsError::InsufficientData`] if `values` is empty.
///
/// # Example
/// ```
/// use statscore_fuzzy::sets::TriangularFuzzyNumber;
/// use statscore_fuzzy::statistics::fuzzy_mean;
///
/// let data = [
///     TriangularFuzzyNumber::new(1.0, 2.0, 3.0).unwrap(),
///     TriangularFuzzyNumber::new(2.0, 3.0, 4.0).unwrap(),
/// ];
/// let mean = fuzzy_mean(&data).unwrap();
/// assert!((mean.m - 2.5).abs() < 1e-12);
/// ```
pub fn fuzzy_mean(values: &[TriangularFuzzyNumber]) -> Result<TriangularFuzzyNumber> {
    if values.is_empty() {
        return Err(StatsError::insufficient_data(1, 0));
    }
    let n = values.len() as f64;
    let a = values.iter().map(|f| f.a).sum::<f64>() / n;
    let m = values.iter().map(|f| f.m).sum::<f64>() / n;
    let b = values.iter().map(|f| f.b).sum::<f64>() / n;

    // Vertices can collide when inputs are (near-)crisp; nudge to keep the
    // strict a < m < b invariant while preserving the centroid.
    if a < m && m < b {
        TriangularFuzzyNumber::new(a, m, b)
    } else {
        let eps = 1e-9_f64.max(m.abs() * 1e-12);
        TriangularFuzzyNumber::new(m - eps, m, m + eps)
    }
}

/// Population variance of the defuzzified (center-of-gravity) values.
///
/// # Errors
/// Returns [`StatsError::InsufficientData`] if `values` is empty.
pub fn fuzzy_variance(values: &[TriangularFuzzyNumber]) -> Result<f64> {
    if values.is_empty() {
        return Err(StatsError::insufficient_data(1, 0));
    }
    let crisp: Vec<f64> = values.iter().map(FuzzyNumber::defuzzify_cog).collect();
    let n = crisp.len() as f64;
    let mean = crisp.iter().sum::<f64>() / n;
    Ok(crisp.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n)
}

/// Pearson correlation of the defuzzified values of two equal-length fuzzy
/// datasets.
///
/// # Errors
/// - [`StatsError::DimensionMismatch`] if the datasets differ in length.
/// - [`StatsError::InsufficientData`] if fewer than two pairs are supplied.
pub fn fuzzy_correlation(
    x: &[TriangularFuzzyNumber],
    y: &[TriangularFuzzyNumber],
) -> Result<f64> {
    if x.len() != y.len() {
        return Err(StatsError::dim_mismatch(format!(
            "fuzzy_correlation: datasets differ in length ({} vs {})",
            x.len(),
            y.len()
        )));
    }
    if x.len() < 2 {
        return Err(StatsError::insufficient_data(2, x.len()));
    }

    let xd: Vec<f64> = x.iter().map(FuzzyNumber::defuzzify_cog).collect();
    let yd: Vec<f64> = y.iter().map(FuzzyNumber::defuzzify_cog).collect();
    let n = xd.len() as f64;
    let mx = xd.iter().sum::<f64>() / n;
    let my = yd.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for (&xi, &yi) in xd.iter().zip(&yd) {
        cov += (xi - mx) * (yi - my);
        vx += (xi - mx).powi(2);
        vy += (yi - my).powi(2);
    }

    if vx == 0.0 || vy == 0.0 {
        return Ok(0.0);
    }
    Ok(cov / (vx * vy).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri(a: f64, m: f64, b: f64) -> TriangularFuzzyNumber {
        TriangularFuzzyNumber::new(a, m, b).unwrap()
    }

    #[test]
    fn mean_vertexwise() {
        let data = [tri(1.0, 2.0, 3.0), tri(2.0, 3.0, 4.0)];
        let mean = fuzzy_mean(&data).unwrap();
        assert!((mean.a - 1.5).abs() < 1e-12);
        assert!((mean.m - 2.5).abs() < 1e-12);
        assert!((mean.b - 3.5).abs() < 1e-12);
    }

    #[test]
    fn mean_empty_errors() {
        assert!(fuzzy_mean(&[]).is_err());
    }

    #[test]
    fn variance_matches_classical_on_crisp() {
        // Symmetric triangles: centroid == peak, so this equals population
        // variance of [1, 2, 3, 4] = 1.25.
        let data = [
            tri(0.5, 1.0, 1.5),
            tri(1.5, 2.0, 2.5),
            tri(2.5, 3.0, 3.5),
            tri(3.5, 4.0, 4.5),
        ];
        let v = fuzzy_variance(&data).unwrap();
        assert!((v - 1.25).abs() < 1e-12);
    }

    #[test]
    fn correlation_perfect_positive() {
        let x = [tri(1.0, 2.0, 3.0), tri(3.0, 4.0, 5.0), tri(5.0, 6.0, 7.0)];
        let y = [tri(2.0, 3.0, 4.0), tri(4.0, 5.0, 6.0), tri(6.0, 7.0, 8.0)];
        let r = fuzzy_correlation(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn correlation_length_mismatch() {
        let x = [tri(1.0, 2.0, 3.0)];
        let y = [tri(1.0, 2.0, 3.0), tri(2.0, 3.0, 4.0)];
        assert!(fuzzy_correlation(&x, &y).is_err());
    }
}
