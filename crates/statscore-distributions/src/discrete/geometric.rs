//! Geometric distribution.

use rand::Rng;
use rand_distr::Distribution as RandDistribution;
use statscore_common::{DiscreteDistribution, Result};

use crate::util::{require_pos, require_prob};

/// Geometric(p) — number of failures before first success (support `{0, 1, 2, …}`).
///
/// PMF: `(1−p)^k · p`. Mean = `(1−p)/p`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometric {
    /// Success probability p ∈ (0, 1].
    pub p: f64,
}

impl Geometric {
    /// Create Geometric(p).
    ///
    /// # Errors
    /// Returns [`StatsError::Domain`] if `p` is outside `(0, 1]`.
    ///
    /// # Example
    /// ```
    /// use statscore_common::DiscreteDistribution;
    /// use statscore_distributions::Geometric;
    ///
    /// let g = Geometric::new(0.5).unwrap();
    /// assert!((g.mean() - 1.0).abs() < 1e-12);
    /// ```
    pub fn new(p: f64) -> Result<Self> {
        require_prob(p)?;
        require_pos(p, "p")?; // p > 0; already ≤ 1
        Ok(Self { p })
    }
}

impl DiscreteDistribution for Geometric {
    fn pmf(&self, k: i64) -> f64 {
        self.log_pmf(k).exp()
    }

    fn log_pmf(&self, k: i64) -> f64 {
        if k < 0 {
            return f64::NEG_INFINITY;
        }
        if self.p == 1.0 {
            return if k == 0 { 0.0 } else { f64::NEG_INFINITY };
        }
        (k as f64) * (1.0 - self.p).ln() + self.p.ln()
    }

    fn cdf(&self, k: i64) -> f64 {
        if k < 0 {
            0.0
        } else if self.p == 1.0 {
            1.0
        } else {
            // 1 − (1−p)^{k+1} = −expm1((k+1) ln(1−p))
            -(((k + 1) as f64) * (1.0 - self.p).ln()).exp_m1()
        }
    }

    fn ppf(&self, p: f64) -> Result<i64> {
        require_prob(p)?;
        if p == 0.0 {
            return Ok(0);
        }
        if p == 1.0 {
            return Ok(i64::MAX);
        }
        if self.p == 1.0 {
            return Ok(0);
        }
        // k = ceil(ln(1−p) / ln(1−success_p)) − 1
        let k = ((1.0 - p).ln() / (1.0 - self.p).ln()).ceil() as i64 - 1;
        Ok(k.max(0))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<i64> {
        // rand_distr Geometric is number of trials until first success (≥ 1).
        // We want failures before first success = trials − 1.
        let dist = rand_distr::Geometric::new(self.p).expect("p validated at construction");
        dist.sample_iter(rng)
            .take(n)
            .map(|trials| trials as i64) // rand 0.6 Geometric: number of failures before success
            .collect()
    }

    fn mean(&self) -> f64 {
        (1.0 - self.p) / self.p
    }

    fn variance(&self) -> f64 {
        (1.0 - self.p) / (self.p * self.p)
    }

    fn skewness(&self) -> f64 {
        (2.0 - self.p) / (1.0 - self.p).sqrt()
    }

    fn kurtosis(&self) -> f64 {
        6.0 + self.p * self.p / (1.0 - self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn basic() {
        let g = Geometric::new(0.5).unwrap();
        assert_relative_eq!(g.pmf(0), 0.5, epsilon = 1e-12);
        assert_relative_eq!(g.pmf(1), 0.25, epsilon = 1e-12);
        assert_relative_eq!(g.mean(), 1.0, epsilon = 1e-12);
    }
}
