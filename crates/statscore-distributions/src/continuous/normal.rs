//! Normal (Gaussian) distribution.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use rand::Rng;
use rand_distr::Distribution as RandDistribution;
use statscore_common::{ContinuousDistribution, Result, StatsError};
use statscore_special::{erf, erf_inv};

use crate::util::{require_pos, require_prob};

/// Normal distribution N(μ, σ²).
///
/// PDF: `(1 / (σ √(2π))) exp(−(x−μ)² / (2σ²))`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal {
    /// Location (mean).
    pub loc: f64,
    /// Scale (standard deviation), must be positive.
    pub scale: f64,
}

impl Normal {
    /// Create a Normal(μ, σ) distribution.
    ///
    /// # Errors
    /// Returns [`StatsError::Domain`] if `scale ≤ 0`.
    ///
    /// # Example
    /// ```
    /// use statscore_common::ContinuousDistribution;
    /// use statscore_distributions::Normal;
    ///
    /// let n = Normal::new(0.0, 1.0).unwrap();
    /// assert!((n.cdf(0.0) - 0.5).abs() < 1e-12);
    /// ```
    pub fn new(loc: f64, scale: f64) -> Result<Self> {
        require_pos(scale, "scale")?;
        if !loc.is_finite() {
            return Err(StatsError::domain("loc must be finite"));
        }
        Ok(Self { loc, scale })
    }

    /// Standard normal N(0, 1).
    #[must_use]
    pub fn standard() -> Self {
        Self {
            loc: 0.0,
            scale: 1.0,
        }
    }
}

impl ContinuousDistribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 0.0;
        }
        let z = (x - self.loc) / self.scale;
        (-0.5 * z * z).exp() / (self.scale * (2.0 * PI).sqrt())
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NEG_INFINITY;
        }
        let z = (x - self.loc) / self.scale;
        -0.5 * z * z - self.scale.ln() - 0.5 * (2.0 * PI).ln()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x == f64::NEG_INFINITY {
            return 0.0;
        }
        if x == f64::INFINITY {
            return 1.0;
        }
        let z = (x - self.loc) / self.scale;
        0.5 * (1.0 + erf(z * FRAC_1_SQRT_2))
    }

    fn ppf(&self, p: f64) -> Result<f64> {
        require_prob(p)?;
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        // Φ⁻¹(p) = √2 erfinv(2p − 1)
        Ok(self.loc + self.scale * std::f64::consts::SQRT_2 * erf_inv(2.0 * p - 1.0))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<f64> {
        let dist =
            rand_distr::Normal::new(self.loc, self.scale).expect("scale validated at construction");
        dist.sample_iter(rng).take(n).collect()
    }

    fn mean(&self) -> f64 {
        self.loc
    }

    fn variance(&self) -> f64 {
        self.scale * self.scale
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn kurtosis(&self) -> f64 {
        0.0
    }

    fn entropy(&self) -> f64 {
        0.5 * (1.0 + (2.0 * PI).ln()) + self.scale.ln()
    }

    fn mode(&self) -> f64 {
        self.loc
    }

    fn median(&self) -> Result<f64> {
        Ok(self.loc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn standard_values() {
        let n = Normal::standard();
        assert_relative_eq!(n.pdf(0.0), 1.0 / (2.0 * PI).sqrt(), max_relative = 1e-12);
        assert_relative_eq!(n.cdf(0.0), 0.5, epsilon = 1e-12);
        assert_relative_eq!(n.ppf(0.5).unwrap(), 0.0, epsilon = 1e-12);
        // SciPy: norm.cdf(1.96) ≈ 0.9750021048517796
        assert_relative_eq!(n.cdf(1.96), 0.975_002_104_851_779_6, max_relative = 1e-10);
    }

    #[test]
    fn cdf_ppf_roundtrip() {
        let n = Normal::new(2.0, 3.0).unwrap();
        for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let x = n.ppf(p).unwrap();
            assert_relative_eq!(n.cdf(x), p, epsilon = 1e-10);
        }
    }

    #[test]
    fn rejects_nonpositive_scale() {
        assert!(Normal::new(0.0, 0.0).is_err());
        assert!(Normal::new(0.0, -1.0).is_err());
    }
}
