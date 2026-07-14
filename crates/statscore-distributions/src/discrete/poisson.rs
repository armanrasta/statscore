//! Poisson distribution.

use rand::Rng;
use rand_distr::Distribution as RandDistribution;
use statscore_common::{DiscreteDistribution, Result};
use statscore_special::{gammaincc, ln_factorial};

use crate::util::{require_pos, require_prob};

/// Poisson(λ) — count of events in a fixed interval with rate λ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poisson {
    /// Rate λ > 0.
    pub lambda: f64,
}

impl Poisson {
    /// Create Poisson(λ).
    ///
    /// # Errors
    /// Returns [`StatsError::Domain`] if `lambda ≤ 0`.
    ///
    /// # Example
    /// ```
    /// use statscore_common::DiscreteDistribution;
    /// use statscore_distributions::Poisson;
    ///
    /// let p = Poisson::new(3.0).unwrap();
    /// assert!((p.mean() - 3.0).abs() < 1e-12);
    /// ```
    pub fn new(lambda: f64) -> Result<Self> {
        require_pos(lambda, "lambda")?;
        Ok(Self { lambda })
    }
}

impl DiscreteDistribution for Poisson {
    fn pmf(&self, k: i64) -> f64 {
        self.log_pmf(k).exp()
    }

    fn log_pmf(&self, k: i64) -> f64 {
        if k < 0 {
            return f64::NEG_INFINITY;
        }
        let k = k as u64;
        k as f64 * self.lambda.ln() - self.lambda - ln_factorial(k)
    }

    fn cdf(&self, k: i64) -> f64 {
        if k < 0 {
            return 0.0;
        }
        // P(X ≤ k) = Q(k+1, λ) = gammaincc(k+1, λ)
        gammaincc((k + 1) as f64, self.lambda)
    }

    fn ppf(&self, p: f64) -> Result<i64> {
        require_prob(p)?;
        if p == 0.0 {
            return Ok(0);
        }
        if p == 1.0 {
            return Ok(i64::MAX);
        }
        // Rough upper bound via gamma inverse / chebyshev
        let mut hi = (self.lambda + 10.0 * (self.lambda + 1.0).sqrt()).ceil() as i64;
        hi = hi.max(10);
        while self.cdf(hi) < p && hi < i64::MAX / 2 {
            hi = (hi as f64 * 1.5).ceil() as i64;
        }
        let mut lo = 0_i64;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.cdf(mid) >= p {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(lo)
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<i64> {
        let dist = rand_distr::Poisson::new(self.lambda).expect("lambda validated at construction");
        dist.sample_iter(rng).take(n).map(|x| x as i64).collect()
    }

    fn mean(&self) -> f64 {
        self.lambda
    }

    fn variance(&self) -> f64 {
        self.lambda
    }

    fn skewness(&self) -> f64 {
        1.0 / self.lambda.sqrt()
    }

    fn kurtosis(&self) -> f64 {
        1.0 / self.lambda
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn unit_poisson() {
        let p = Poisson::new(1.0).unwrap();
        assert_relative_eq!(p.pmf(0), (-1.0_f64).exp(), epsilon = 1e-12);
        assert_relative_eq!(p.pmf(1), (-1.0_f64).exp(), epsilon = 1e-12);
        assert!(p.cdf(0) > 0.0 && p.cdf(0) < 1.0);
    }

    #[test]
    fn ppf_cdf_consistency() {
        let p = Poisson::new(4.0).unwrap();
        for &q in &[0.1, 0.5, 0.9] {
            let k = p.ppf(q).unwrap();
            assert!(p.cdf(k) >= q);
            if k > 0 {
                assert!(p.cdf(k - 1) < q);
            }
        }
    }
}
