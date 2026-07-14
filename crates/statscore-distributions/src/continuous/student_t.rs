//! Student's t distribution.

use std::f64::consts::PI;

use rand::Rng;
use rand_distr::Distribution as RandDistribution;
use statscore_common::{ContinuousDistribution, Result};
use statscore_special::{betainc, betaincinv, digamma, ln_beta, ln_gamma};

use crate::util::{require_pos, require_prob};

/// Student's t distribution with ν degrees of freedom (location 0, scale 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentT {
    /// Degrees of freedom ν > 0.
    pub df: f64,
}

impl StudentT {
    /// Create t(ν).
    ///
    /// # Errors
    /// Returns [`StatsError::Domain`] if `df ≤ 0`.
    ///
    /// # Example
    /// ```
    /// use statscore_common::ContinuousDistribution;
    /// use statscore_distributions::StudentT;
    ///
    /// let t = StudentT::new(10.0).unwrap();
    /// assert!((t.cdf(0.0) - 0.5).abs() < 1e-12);
    /// ```
    pub fn new(df: f64) -> Result<Self> {
        require_pos(df, "df")?;
        Ok(Self { df })
    }
}

impl ContinuousDistribution for StudentT {
    fn pdf(&self, x: f64) -> f64 {
        self.log_pdf(x).exp()
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NEG_INFINITY;
        }
        let nu = self.df;
        ln_gamma((nu + 1.0) / 2.0)
            - ln_gamma(nu / 2.0)
            - 0.5 * (nu * PI).ln()
            - ((nu + 1.0) / 2.0) * (1.0 + x * x / nu).ln()
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
        let nu = self.df;
        let z = nu / (nu + x * x);
        let incomplete = 0.5 * betainc(nu / 2.0, 0.5, z);
        if x >= 0.0 {
            1.0 - incomplete
        } else {
            incomplete
        }
    }

    fn ppf(&self, p: f64) -> Result<f64> {
        require_prob(p)?;
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        if (p - 0.5).abs() < 1e-16 {
            return Ok(0.0);
        }
        let nu = self.df;
        let p_tail = if p >= 0.5 { 2.0 * (1.0 - p) } else { 2.0 * p };
        let z = betaincinv(nu / 2.0, 0.5, p_tail);
        let x = (nu * (1.0 - z) / z).sqrt();
        Ok(if p >= 0.5 { x } else { -x })
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<f64> {
        let dist = rand_distr::StudentT::new(self.df).expect("df validated at construction");
        dist.sample_iter(rng).take(n).collect()
    }

    fn mean(&self) -> f64 {
        if self.df > 1.0 { 0.0 } else { f64::NAN }
    }

    fn variance(&self) -> f64 {
        if self.df > 2.0 {
            self.df / (self.df - 2.0)
        } else {
            f64::NAN
        }
    }

    fn skewness(&self) -> f64 {
        if self.df > 3.0 { 0.0 } else { f64::NAN }
    }

    fn kurtosis(&self) -> f64 {
        if self.df > 4.0 {
            6.0 / (self.df - 4.0)
        } else {
            f64::NAN
        }
    }

    fn entropy(&self) -> f64 {
        // H = (ν+1)/2 · (ψ((ν+1)/2) − ψ(ν/2)) + ln(√ν · B(ν/2, 1/2))
        let nu = self.df;
        0.5 * (nu + 1.0) * (digamma((nu + 1.0) / 2.0) - digamma(nu / 2.0))
            + 0.5 * nu.ln()
            + ln_beta(nu / 2.0, 0.5)
    }

    fn mode(&self) -> f64 {
        0.0
    }

    fn median(&self) -> Result<f64> {
        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn symmetric_at_zero() {
        let t = StudentT::new(5.0).unwrap();
        assert_relative_eq!(t.cdf(0.0), 0.5, epsilon = 1e-12);
        assert_relative_eq!(t.pdf(1.0), t.pdf(-1.0), epsilon = 1e-12);
    }

    #[test]
    fn roundtrip() {
        let t = StudentT::new(8.0).unwrap();
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            assert_relative_eq!(t.cdf(t.ppf(p).unwrap()), p, epsilon = 1e-8);
        }
    }
}
