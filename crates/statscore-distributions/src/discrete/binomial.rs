//! Binomial distribution.

use rand::Rng;
use rand_distr::Distribution as RandDistribution;
use statscore_common::{DiscreteDistribution, Result};
use statscore_special::{betainc, ln_choose};

use crate::util::require_prob;

/// Binomial(n, p) — number of successes in `n` independent Bernoulli(`p`) trials.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Binomial {
    /// Number of trials (n ≥ 0).
    pub n: u64,
    /// Success probability p ∈ [0, 1].
    pub p: f64,
}

impl Binomial {
    /// Create Binomial(n, p).
    ///
    /// # Errors
    /// Returns [`StatsError`] if `p` is outside `[0, 1]`.
    ///
    /// # Example
    /// ```
    /// use statscore_common::DiscreteDistribution;
    /// use statscore_distributions::Binomial;
    ///
    /// let b = Binomial::new(10, 0.5).unwrap();
    /// assert!((b.mean() - 5.0).abs() < 1e-12);
    /// ```
    pub fn new(n: u64, p: f64) -> Result<Self> {
        require_prob(p)?;
        Ok(Self { n, p })
    }
}

impl DiscreteDistribution for Binomial {
    fn pmf(&self, k: i64) -> f64 {
        self.log_pmf(k).exp()
    }

    fn log_pmf(&self, k: i64) -> f64 {
        if k < 0 || k as u64 > self.n {
            return f64::NEG_INFINITY;
        }
        let k = k as u64;
        if self.p == 0.0 {
            return if k == 0 { 0.0 } else { f64::NEG_INFINITY };
        }
        if self.p == 1.0 {
            return if k == self.n { 0.0 } else { f64::NEG_INFINITY };
        }
        ln_choose(self.n, k)
            + (k as f64) * self.p.ln()
            + ((self.n - k) as f64) * (1.0 - self.p).ln()
    }

    fn cdf(&self, k: i64) -> f64 {
        if k < 0 {
            return 0.0;
        }
        if k as u64 >= self.n {
            return 1.0;
        }
        // P(X ≤ k) = 1 − I_p(k+1, n−k) = I_{1−p}(n−k, k+1)
        let k = k as u64;
        betainc((self.n - k) as f64, (k + 1) as f64, 1.0 - self.p)
    }

    fn ppf(&self, p: f64) -> Result<i64> {
        require_prob(p)?;
        if p == 0.0 {
            return Ok(0);
        }
        if p == 1.0 {
            return Ok(self.n as i64);
        }
        // Binary search for smallest k with cdf(k) ≥ p
        let mut lo = 0_i64;
        let mut hi = self.n as i64;
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
        let dist = rand_distr::Binomial::new(self.n, self.p)
            .expect("parameters validated at construction");
        dist.sample_iter(rng).take(n).map(|x| x as i64).collect()
    }

    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }

    fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }

    fn skewness(&self) -> f64 {
        let n = self.n as f64;
        (1.0 - 2.0 * self.p) / (n * self.p * (1.0 - self.p)).sqrt()
    }

    fn kurtosis(&self) -> f64 {
        let n = self.n as f64;
        (1.0 - 6.0 * self.p * (1.0 - self.p)) / (n * self.p * (1.0 - self.p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn fair_coin() {
        let b = Binomial::new(10, 0.5).unwrap();
        assert_relative_eq!(b.mean(), 5.0, epsilon = 1e-12);
        assert_relative_eq!(b.pmf(5), 252.0 / 1024.0, max_relative = 1e-12);
        assert!(b.cdf(10) == 1.0);
        assert_eq!(b.ppf(0.5).unwrap(), 5);
    }
}
