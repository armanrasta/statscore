//! Gamma distribution (shape–scale parameterization).

use rand::Rng;
use rand_distr::Distribution as RandDistribution;
use statscore_common::{ContinuousDistribution, Result};
use statscore_special::{digamma, gammainc, gammaincc, ln_gamma};

use crate::util::{gammaincinv, require_pos, require_prob};

/// Gamma(α, θ) with shape `α` and scale `θ`.
///
/// PDF: `x^{α−1} e^{−x/θ} / (θ^α Γ(α))` for `x > 0`.
/// Mean = `α θ`, variance = `α θ²`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gamma {
    /// Shape α > 0.
    pub shape: f64,
    /// Scale θ > 0.
    pub scale: f64,
}

impl Gamma {
    /// Create Gamma(shape, scale).
    ///
    /// # Errors
    /// Returns [`StatsError::Domain`] if shape or scale ≤ 0.
    ///
    /// # Example
    /// ```
    /// use statscore_common::ContinuousDistribution;
    /// use statscore_distributions::Gamma;
    ///
    /// let g = Gamma::new(2.0, 1.0).unwrap();
    /// assert!((g.mean() - 2.0).abs() < 1e-12);
    /// ```
    pub fn new(shape: f64, scale: f64) -> Result<Self> {
        require_pos(shape, "shape")?;
        require_pos(scale, "scale")?;
        Ok(Self { shape, scale })
    }
}

impl ContinuousDistribution for Gamma {
    fn pdf(&self, x: f64) -> f64 {
        self.log_pdf(x).exp()
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if x < 0.0 || !x.is_finite() {
            return f64::NEG_INFINITY;
        }
        if x == 0.0 {
            return if self.shape == 1.0 {
                -self.scale.ln()
            } else if self.shape < 1.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
        }
        (self.shape - 1.0) * x.ln()
            - x / self.scale
            - self.shape * self.scale.ln()
            - ln_gamma(self.shape)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            gammainc(self.shape, x / self.scale)
        }
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            1.0
        } else {
            gammaincc(self.shape, x / self.scale)
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
        Ok(self.scale * gammaincinv(self.shape, p))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<f64> {
        let dist = rand_distr::Gamma::new(self.shape, self.scale)
            .expect("parameters validated at construction");
        dist.sample_iter(rng).take(n).collect()
    }

    fn mean(&self) -> f64 {
        self.shape * self.scale
    }

    fn variance(&self) -> f64 {
        self.shape * self.scale * self.scale
    }

    fn skewness(&self) -> f64 {
        2.0 / self.shape.sqrt()
    }

    fn kurtosis(&self) -> f64 {
        6.0 / self.shape
    }

    fn entropy(&self) -> f64 {
        self.shape
            + self.scale.ln()
            + ln_gamma(self.shape)
            + (1.0 - self.shape) * digamma(self.shape)
    }

    fn mode(&self) -> f64 {
        if self.shape >= 1.0 {
            (self.shape - 1.0) * self.scale
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
    fn exponential_special_case() {
        // Gamma(1, 1/λ) = Exp(λ)
        let g = Gamma::new(1.0, 0.5).unwrap(); // mean = 0.5 → rate 2
        assert_relative_eq!(g.cdf(1.0), 1.0 - (-2.0_f64).exp(), epsilon = 1e-10);
    }

    #[test]
    fn roundtrip() {
        let g = Gamma::new(2.5, 1.5).unwrap();
        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            assert_relative_eq!(g.cdf(g.ppf(p).unwrap()), p, epsilon = 1e-8);
        }
    }
}
