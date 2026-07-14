//! Fisher–Snedecor F distribution.

use rand::Rng;
use rand_distr::Distribution as RandDistribution;
use statscore_common::{ContinuousDistribution, Result};
use statscore_special::{betainc, betaincinv, ln_beta};

use crate::util::{require_pos, require_prob};

/// F(d₁, d₂) distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FDistribution {
    /// Numerator degrees of freedom d₁ > 0.
    pub dfn: f64,
    /// Denominator degrees of freedom d₂ > 0.
    pub dfd: f64,
}

impl FDistribution {
    /// Create F(d₁, d₂).
    ///
    /// # Errors
    /// Returns [`StatsError::Domain`] if either df ≤ 0.
    ///
    /// # Example
    /// ```
    /// use statscore_common::ContinuousDistribution;
    /// use statscore_distributions::FDistribution;
    ///
    /// let f = FDistribution::new(5.0, 10.0).unwrap();
    /// assert!(f.cdf(1.0) > 0.0 && f.cdf(1.0) < 1.0);
    /// ```
    pub fn new(dfn: f64, dfd: f64) -> Result<Self> {
        require_pos(dfn, "dfn")?;
        require_pos(dfd, "dfd")?;
        Ok(Self { dfn, dfd })
    }
}

impl ContinuousDistribution for FDistribution {
    fn pdf(&self, x: f64) -> f64 {
        self.log_pdf(x).exp()
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if x < 0.0 || !x.is_finite() {
            return f64::NEG_INFINITY;
        }
        if x == 0.0 {
            return if self.dfn < 2.0 {
                f64::INFINITY
            } else if self.dfn == 2.0 {
                // finite
                let d1 = self.dfn;
                let d2 = self.dfd;
                0.5 * d1 * (d1 / d2).ln() - ln_beta(d1 / 2.0, d2 / 2.0)
            } else {
                f64::NEG_INFINITY
            };
        }
        let d1 = self.dfn;
        let d2 = self.dfd;
        0.5 * d1 * (d1 / d2).ln() + (0.5 * d1 - 1.0) * x.ln()
            - 0.5 * (d1 + d2) * (1.0 + d1 * x / d2).ln()
            - ln_beta(d1 / 2.0, d2 / 2.0)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x.is_infinite() {
            return 1.0;
        }
        let d1 = self.dfn;
        let d2 = self.dfd;
        let z = (d1 * x) / (d1 * x + d2);
        betainc(d1 / 2.0, d2 / 2.0, z)
    }

    fn ppf(&self, p: f64) -> Result<f64> {
        require_prob(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        let d1 = self.dfn;
        let d2 = self.dfd;
        let z = betaincinv(d1 / 2.0, d2 / 2.0, p);
        // z = d1 x / (d1 x + d2) ⇒ x = (d2 / d1) · z / (1 − z)
        Ok((d2 / d1) * z / (1.0 - z))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<f64> {
        let dist = rand_distr::FisherF::new(self.dfn, self.dfd)
            .expect("parameters validated at construction");
        dist.sample_iter(rng).take(n).collect()
    }

    fn mean(&self) -> f64 {
        if self.dfd > 2.0 {
            self.dfd / (self.dfd - 2.0)
        } else {
            f64::NAN
        }
    }

    fn variance(&self) -> f64 {
        let d1 = self.dfn;
        let d2 = self.dfd;
        if d2 > 4.0 {
            2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0) * (d2 - 2.0) * (d2 - 4.0))
        } else {
            f64::NAN
        }
    }

    fn skewness(&self) -> f64 {
        let d1 = self.dfn;
        let d2 = self.dfd;
        if d2 > 6.0 {
            (2.0 * d1 + d2 - 2.0) * (8.0 * (d2 - 4.0)).sqrt()
                / ((d2 - 6.0) * (d1 * (d1 + d2 - 2.0)).sqrt())
        } else {
            f64::NAN
        }
    }

    fn kurtosis(&self) -> f64 {
        f64::NAN
    }

    fn entropy(&self) -> f64 {
        f64::NAN
    }

    fn mode(&self) -> f64 {
        if self.dfn > 2.0 {
            (self.dfn - 2.0) / self.dfn * self.dfd / (self.dfd + 2.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cdf_in_unit_interval() {
        let f = FDistribution::new(5.0, 10.0).unwrap();
        let c = f.cdf(1.5);
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn roundtrip() {
        let f = FDistribution::new(4.0, 12.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            assert_relative_eq!(f.cdf(f.ppf(p).unwrap()), p, epsilon = 1e-7);
        }
    }
}
