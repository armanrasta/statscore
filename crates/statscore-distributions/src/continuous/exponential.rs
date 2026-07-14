//! Exponential distribution.

use rand::Rng;
use rand_distr::Distribution as RandDistribution;
use statscore_common::{ContinuousDistribution, Result};

use crate::util::{require_pos, require_prob};

/// Exponential distribution with rate parameter λ > 0.
///
/// PDF: `λ e^{−λx}` for `x ≥ 0`. Mean = `1/λ`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exponential {
    /// Rate parameter λ.
    pub rate: f64,
}

impl Exponential {
    /// Create Exp(λ).
    ///
    /// # Errors
    /// Returns [`StatsError::Domain`] if `rate ≤ 0`.
    ///
    /// # Example
    /// ```
    /// use statscore_common::ContinuousDistribution;
    /// use statscore_distributions::Exponential;
    ///
    /// let e = Exponential::new(2.0).unwrap();
    /// assert!((e.mean() - 0.5).abs() < 1e-12);
    /// ```
    pub fn new(rate: f64) -> Result<Self> {
        require_pos(rate, "rate")?;
        Ok(Self { rate })
    }

    /// Create Exp with given mean (scale = 1/rate).
    pub fn from_mean(mean: f64) -> Result<Self> {
        require_pos(mean, "mean")?;
        Ok(Self { rate: 1.0 / mean })
    }
}

impl ContinuousDistribution for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.rate * (-self.rate * x).exp()
        }
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else {
            self.rate.ln() - self.rate * x
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            -(-self.rate * x).exp_m1() // 1 - e^{−λx}
        }
    }

    fn sf(&self, x: f64) -> f64 {
        if x < 0.0 { 1.0 } else { (-self.rate * x).exp() }
    }

    fn ppf(&self, p: f64) -> Result<f64> {
        require_prob(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        // −ln(1−p) / λ; use log1p for p near 0, log1mexp style for p near 1
        Ok(-(1.0 - p).ln() / self.rate)
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<f64> {
        let dist = rand_distr::Exp::new(self.rate).expect("rate validated at construction");
        dist.sample_iter(rng).take(n).collect()
    }

    fn mean(&self) -> f64 {
        1.0 / self.rate
    }

    fn variance(&self) -> f64 {
        1.0 / (self.rate * self.rate)
    }

    fn skewness(&self) -> f64 {
        2.0
    }

    fn kurtosis(&self) -> f64 {
        6.0
    }

    fn entropy(&self) -> f64 {
        1.0 - self.rate.ln()
    }

    fn mode(&self) -> f64 {
        0.0
    }

    fn median(&self) -> Result<f64> {
        Ok(std::f64::consts::LN_2 / self.rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn unit_rate() {
        let e = Exponential::new(1.0).unwrap();
        assert_relative_eq!(e.cdf(1.0), 1.0 - (-1.0_f64).exp(), epsilon = 1e-12);
        assert_relative_eq!(e.ppf(0.5).unwrap(), std::f64::consts::LN_2, epsilon = 1e-12);
        assert_relative_eq!(e.sf(1.0), (-1.0_f64).exp(), epsilon = 1e-12);
    }

    #[test]
    fn roundtrip() {
        let e = Exponential::new(3.5).unwrap();
        for &p in &[0.01, 0.2, 0.5, 0.8, 0.99] {
            assert_relative_eq!(e.cdf(e.ppf(p).unwrap()), p, epsilon = 1e-10);
        }
    }
}
