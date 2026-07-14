//! Chi-squared distribution.

use rand::Rng;
use rand_distr::Distribution as RandDistribution;
use statscore_common::{ContinuousDistribution, Result};
use statscore_special::{digamma, gammainc, gammaincc, ln_gamma};

use crate::util::{gammaincinv, require_pos, require_prob};

/// χ²(ν) with ν degrees of freedom.
///
/// Equivalent to Gamma(ν/2, scale = 2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquared {
    /// Degrees of freedom ν > 0.
    pub df: f64,
}

impl ChiSquared {
    /// Create χ²(ν).
    ///
    /// # Errors
    /// Returns [`StatsError::Domain`] if `df ≤ 0`.
    ///
    /// # Example
    /// ```
    /// use statscore_common::ContinuousDistribution;
    /// use statscore_distributions::ChiSquared;
    ///
    /// let c = ChiSquared::new(2.0).unwrap();
    /// assert!((c.mean() - 2.0).abs() < 1e-12);
    /// ```
    pub fn new(df: f64) -> Result<Self> {
        require_pos(df, "df")?;
        Ok(Self { df })
    }
}

impl ContinuousDistribution for ChiSquared {
    fn pdf(&self, x: f64) -> f64 {
        self.log_pdf(x).exp()
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if x < 0.0 || !x.is_finite() {
            return f64::NEG_INFINITY;
        }
        let k = self.df / 2.0;
        if x == 0.0 {
            return if k == 1.0 {
                -std::f64::consts::LN_2
            } else if k < 1.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
        }
        (k - 1.0) * x.ln() - x / 2.0 - k * std::f64::consts::LN_2 - ln_gamma(k)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            gammainc(self.df / 2.0, x / 2.0)
        }
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            1.0
        } else {
            gammaincc(self.df / 2.0, x / 2.0)
        }
    }

    fn ppf(&self, p: f64) -> Result<f64> {
        require_prob(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(2.0 * gammaincinv(self.df / 2.0, p))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<f64> {
        let dist = rand_distr::ChiSquared::new(self.df).expect("df validated at construction");
        dist.sample_iter(rng).take(n).collect()
    }

    fn mean(&self) -> f64 {
        self.df
    }

    fn variance(&self) -> f64 {
        2.0 * self.df
    }

    fn skewness(&self) -> f64 {
        (8.0 / self.df).sqrt()
    }

    fn kurtosis(&self) -> f64 {
        12.0 / self.df
    }

    fn entropy(&self) -> f64 {
        // H = ν/2 + ln(2 Γ(ν/2)) + (1 − ν/2) ψ(ν/2)
        let k = self.df / 2.0;
        k + std::f64::consts::LN_2 + ln_gamma(k) + (1.0 - k) * digamma(k)
    }

    fn mode(&self) -> f64 {
        (self.df - 2.0).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn df_two_matches_exp() {
        let c = ChiSquared::new(2.0).unwrap();
        assert_relative_eq!(c.cdf(2.0), 1.0 - (-1.0_f64).exp(), epsilon = 1e-10);
    }

    #[test]
    fn roundtrip() {
        let c = ChiSquared::new(5.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            assert_relative_eq!(c.cdf(c.ppf(p).unwrap()), p, epsilon = 1e-8);
        }
    }
}
