//! Beta distribution.

use rand::Rng;
use rand_distr::Distribution as RandDistribution;
use statscore_common::{ContinuousDistribution, Result};
use statscore_special::{betainc, betaincinv, digamma, ln_beta};

use crate::util::{require_pos, require_prob};

/// Beta(α, β) distribution on (0, 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beta {
    /// Shape α > 0.
    pub alpha: f64,
    /// Shape β > 0.
    pub beta: f64,
}

impl Beta {
    /// Create Beta(α, β).
    ///
    /// # Errors
    /// Returns [`StatsError::Domain`] if α or β ≤ 0.
    ///
    /// # Example
    /// ```
    /// use statscore_common::ContinuousDistribution;
    /// use statscore_distributions::Beta;
    ///
    /// let b = Beta::new(2.0, 5.0).unwrap();
    /// assert!((b.mean() - 2.0 / 7.0).abs() < 1e-12);
    /// ```
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        require_pos(alpha, "alpha")?;
        require_pos(beta, "beta")?;
        Ok(Self { alpha, beta })
    }
}

impl ContinuousDistribution for Beta {
    fn pdf(&self, x: f64) -> f64 {
        self.log_pdf(x).exp()
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if x < 0.0 || x > 1.0 {
            return f64::NEG_INFINITY;
        }
        if x == 0.0 || x == 1.0 {
            // Boundary: zero unless the corresponding shape ≤ 1
            if (x == 0.0 && self.alpha < 1.0) || (x == 1.0 && self.beta < 1.0) {
                return f64::INFINITY;
            }
            if (x == 0.0 && self.alpha == 1.0) || (x == 1.0 && self.beta == 1.0) {
                return -ln_beta(self.alpha, self.beta);
            }
            return f64::NEG_INFINITY;
        }
        (self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - x).ln()
            - ln_beta(self.alpha, self.beta)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            betainc(self.alpha, self.beta, x)
        }
    }

    fn ppf(&self, p: f64) -> Result<f64> {
        require_prob(p)?;
        Ok(betaincinv(self.alpha, self.beta, p))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<f64> {
        let dist = rand_distr::Beta::new(self.alpha, self.beta)
            .expect("parameters validated at construction");
        dist.sample_iter(rng).take(n).collect()
    }

    fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    fn variance(&self) -> f64 {
        let s = self.alpha + self.beta;
        self.alpha * self.beta / (s * s * (s + 1.0))
    }

    fn skewness(&self) -> f64 {
        let s = self.alpha + self.beta;
        2.0 * (self.beta - self.alpha) * (s + 1.0).sqrt()
            / ((s + 2.0) * (self.alpha * self.beta).sqrt())
    }

    fn kurtosis(&self) -> f64 {
        let a = self.alpha;
        let b = self.beta;
        let s = a + b;
        let num = 6.0
            * (a * a * a - a * a * (2.0 * b - 1.0) + b * b * (b + 1.0) - 2.0 * a * b * (b + 2.0));
        let den = a * b * (s + 2.0) * (s + 3.0);
        num / den
    }

    fn entropy(&self) -> f64 {
        let a = self.alpha;
        let b = self.beta;
        ln_beta(a, b) - (a - 1.0) * digamma(a) - (b - 1.0) * digamma(b)
            + (a + b - 2.0) * digamma(a + b)
    }

    fn mode(&self) -> f64 {
        if self.alpha > 1.0 && self.beta > 1.0 {
            (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        } else {
            f64::NAN
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mean_and_cdf() {
        let b = Beta::new(2.0, 5.0).unwrap();
        assert_relative_eq!(b.mean(), 2.0 / 7.0, epsilon = 1e-12);
        // scipy.special.betainc(2, 5, 0.3) = 0.579825
        assert_relative_eq!(b.cdf(0.3), 0.579_825, max_relative = 1e-5);
    }

    #[test]
    fn roundtrip() {
        let b = Beta::new(2.5, 3.5).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            assert_relative_eq!(b.cdf(b.ppf(p).unwrap()), p, epsilon = 1e-8);
        }
    }
}
