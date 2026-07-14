//! Continuous uniform distribution.

use rand::Rng;
use rand_distr::Distribution as RandDistribution;
use statscore_common::{ContinuousDistribution, Result};

use crate::util::{require_interval, require_prob};

/// Continuous Uniform(a, b) on `[a, b]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uniform {
    /// Lower endpoint (inclusive).
    pub a: f64,
    /// Upper endpoint (inclusive).
    pub b: f64,
}

impl Uniform {
    /// Create Uniform(a, b) with `a < b`.
    ///
    /// # Errors
    /// Returns [`StatsError::Domain`] if `a ≥ b` or endpoints are non-finite.
    ///
    /// # Example
    /// ```
    /// use statscore_common::ContinuousDistribution;
    /// use statscore_distributions::Uniform;
    ///
    /// let u = Uniform::new(0.0, 1.0).unwrap();
    /// assert!((u.cdf(0.5) - 0.5).abs() < 1e-12);
    /// ```
    pub fn new(a: f64, b: f64) -> Result<Self> {
        require_interval(a, b)?;
        Ok(Self { a, b })
    }

    #[must_use]
    fn width(&self) -> f64 {
        self.b - self.a
    }
}

impl ContinuousDistribution for Uniform {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b {
            0.0
        } else {
            1.0 / self.width()
        }
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b {
            f64::NEG_INFINITY
        } else {
            -self.width().ln()
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < self.a {
            0.0
        } else if x >= self.b {
            1.0
        } else {
            (x - self.a) / self.width()
        }
    }

    fn ppf(&self, p: f64) -> Result<f64> {
        require_prob(p)?;
        Ok(self.a + p * self.width())
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<f64> {
        let dist =
            rand_distr::Uniform::new(self.a, self.b).expect("interval validated at construction");
        dist.sample_iter(rng).take(n).collect()
    }

    fn mean(&self) -> f64 {
        0.5 * (self.a + self.b)
    }

    fn variance(&self) -> f64 {
        let w = self.width();
        w * w / 12.0
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn kurtosis(&self) -> f64 {
        -6.0 / 5.0
    }

    fn entropy(&self) -> f64 {
        self.width().ln()
    }

    fn mode(&self) -> f64 {
        f64::NAN // any point in [a,b]
    }

    fn median(&self) -> Result<f64> {
        Ok(self.mean())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn unit_interval() {
        let u = Uniform::new(0.0, 1.0).unwrap();
        assert_relative_eq!(u.pdf(0.3), 1.0, epsilon = 1e-15);
        assert_eq!(u.pdf(-0.1), 0.0);
        assert_relative_eq!(u.cdf(0.25), 0.25, epsilon = 1e-15);
        assert_relative_eq!(u.ppf(0.75).unwrap(), 0.75, epsilon = 1e-15);
    }

    #[test]
    fn rejects_bad_interval() {
        assert!(Uniform::new(1.0, 1.0).is_err());
        assert!(Uniform::new(2.0, 1.0).is_err());
    }
}
