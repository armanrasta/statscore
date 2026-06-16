//! Core distribution traits.
//!
//! ## Design decisions
//!
//! ### `log_pdf` returns `f64`, not `Option<f64>`
//! Returns `f64::NEG_INFINITY` for points outside the support.
//! This matches NumPy/SciPy convention and avoids unwrap noise downstream.
//!
//! ### `mean`, `variance`, etc. return `f64`
//! Returns `f64::NAN` when mathematically undefined (e.g., Cauchy mean).
//! Document this clearly. Returning `Option` forces downstream match arms
//! for the common case where the moment is always defined.
//!
//! ### `sample` writes into a `Vec<f64>` returned by value
//! For large samples the allocation is unavoidable anyway.
//! The Python layer maps this directly to a NumPy array.

use rand::Rng;
use crate::error::Result;

// ── Continuous distributions ──────────────────────────────────────────────────

/// A probability distribution over a continuous real-valued random variable.
///
/// Implementors must provide at minimum:
/// - [`pdf`](ContinuousDistribution::pdf)
/// - [`cdf`](ContinuousDistribution::cdf)
/// - [`ppf`](ContinuousDistribution::ppf)
/// - [`sample`](ContinuousDistribution::sample)
///
/// All other methods have correct default implementations derived from these.
pub trait ContinuousDistribution {
    // ── Required ─────────────────────────────────────────────────────────────

    /// Probability density function evaluated at `x`.
    ///
    /// Returns `0.0` for `x` outside the support.
    fn pdf(&self, x: f64) -> f64;

    /// Cumulative distribution function: P(X ≤ x).
    fn cdf(&self, x: f64) -> f64;

    /// Percent-point function (quantile function / inverse CDF).
    ///
    /// Returns the value `x` such that `P(X ≤ x) = p`.
    ///
    /// # Errors
    /// - [`StatsError::Domain`] if `p` is not in `[0, 1]`.
    fn ppf(&self, p: f64) -> Result<f64>;

    /// Draw `n` independent samples using the provided random number generator.
    ///
    /// # Example
    /// ```ignore
    /// let mut rng = rand::thread_rng();
    /// let samples = dist.sample(&mut rng, 1000);
    /// ```
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<f64>;

    // ── Derived (override for performance) ───────────────────────────────────

    /// Natural logarithm of the PDF.
    ///
    /// Returns `f64::NEG_INFINITY` for `x` outside the support (where `pdf == 0`).
    /// Override this in implementations where a direct formula is more stable.
    fn log_pdf(&self, x: f64) -> f64 {
        let p = self.pdf(x);
        if p > 0.0 {
            p.ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Survival function: P(X > x) = 1 − CDF(x).
    ///
    /// Override when a direct formula avoids catastrophic cancellation for large x.
    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    /// Inverse survival function: the value x such that P(X > x) = p.
    fn isf(&self, p: f64) -> Result<f64> {
        self.ppf(1.0 - p)
    }

    /// Hazard function: h(x) = f(x) / S(x).
    ///
    /// Returns `f64::INFINITY` when `sf(x) == 0`.
    fn hazard(&self, x: f64) -> f64 {
        let s = self.sf(x);
        if s > 0.0 {
            self.pdf(x) / s
        } else {
            f64::INFINITY
        }
    }

    /// Cumulative hazard function: H(x) = −ln S(x).
    fn cumulative_hazard(&self, x: f64) -> f64 {
        -self.sf(x).ln()
    }

    // ── Moments ───────────────────────────────────────────────────────────────
    //
    // Return f64::NAN when the moment is mathematically undefined.
    // Document this on every implementation.

    /// Expected value E[X].
    ///
    /// Returns `f64::NAN` if undefined (e.g., Cauchy distribution).
    fn mean(&self) -> f64 {
        f64::NAN
    }

    /// Variance Var(X) = E[(X − μ)²].
    ///
    /// Returns `f64::NAN` if undefined.
    fn variance(&self) -> f64 {
        f64::NAN
    }

    /// Standard deviation: √Var(X).
    ///
    /// Returns `f64::NAN` if variance is undefined.
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Skewness E[(X − μ)³] / σ³.
    ///
    /// Returns `f64::NAN` if undefined.
    fn skewness(&self) -> f64 {
        f64::NAN
    }

    /// Excess kurtosis E[(X − μ)⁴] / σ⁴ − 3.
    ///
    /// Returns `f64::NAN` if undefined.
    fn kurtosis(&self) -> f64 {
        f64::NAN
    }

    /// Differential entropy h(X) = −∫ f(x) ln f(x) dx (in nats).
    ///
    /// Returns `f64::NAN` if not implemented for this distribution.
    fn entropy(&self) -> f64 {
        f64::NAN
    }

    /// Median: the value m such that P(X ≤ m) = 0.5.
    fn median(&self) -> Result<f64> {
        self.ppf(0.5)
    }

    /// Mode: the value where the PDF is maximised.
    ///
    /// Returns `f64::NAN` if undefined or not overridden.
    fn mode(&self) -> f64 {
        f64::NAN
    }
}

// ── Discrete distributions ────────────────────────────────────────────────────

/// A probability distribution over a discrete integer-valued random variable.
///
/// Mirrors [`ContinuousDistribution`] with `pmf` / integer arguments.
pub trait DiscreteDistribution {
    // ── Required ─────────────────────────────────────────────────────────────

    /// Probability mass function: P(X = k).
    ///
    /// Returns `0.0` for k outside the support.
    fn pmf(&self, k: i64) -> f64;

    /// Cumulative distribution function: P(X ≤ k).
    fn cdf(&self, k: i64) -> f64;

    /// Quantile function: smallest k such that P(X ≤ k) ≥ p.
    ///
    /// # Errors
    /// [`StatsError::Domain`] if `p` not in `[0, 1]`.
    fn ppf(&self, p: f64) -> Result<i64>;

    /// Draw `n` independent samples.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<i64>;

    // ── Derived ───────────────────────────────────────────────────────────────

    /// log P(X = k). Returns `f64::NEG_INFINITY` for k outside support.
    fn log_pmf(&self, k: i64) -> f64 {
        let p = self.pmf(k);
        if p > 0.0 { p.ln() } else { f64::NEG_INFINITY }
    }

    /// Survival function: P(X > k) = 1 − P(X ≤ k).
    fn sf(&self, k: i64) -> f64 {
        1.0 - self.cdf(k)
    }

    /// Inverse survival function.
    fn isf(&self, p: f64) -> Result<i64> {
        self.ppf(1.0 - p)
    }

    /// Expected value. Returns `f64::NAN` if undefined.
    fn mean(&self) -> f64 { f64::NAN }

    /// Variance. Returns `f64::NAN` if undefined.
    fn variance(&self) -> f64 { f64::NAN }

    /// Standard deviation. Returns `f64::NAN` if undefined.
    fn std_dev(&self) -> f64 { self.variance().sqrt() }

    /// Skewness. Returns `f64::NAN` if undefined.
    fn skewness(&self) -> f64 { f64::NAN }

    /// Excess kurtosis. Returns `f64::NAN` if undefined.
    fn kurtosis(&self) -> f64 { f64::NAN }

    /// Entropy in nats. Returns `f64::NAN` if not implemented.
    fn entropy(&self) -> f64 { f64::NAN }
}

// ── Fitting traits ────────────────────────────────────────────────────────────

/// Maximum likelihood estimation from observed data.
pub trait MleFit: Sized {
    /// Fit the distribution to `data` via maximum likelihood.
    ///
    /// # Errors
    /// - [`StatsError::InsufficientData`] if not enough observations.
    /// - [`StatsError::Convergence`] if MLE optimisation did not converge.
    fn fit_mle(data: &[f64]) -> Result<Self>;
}

/// Method of moments estimation from observed data.
pub trait MomFit: Sized {
    /// Fit the distribution to `data` via method of moments.
    ///
    /// # Errors
    /// - [`StatsError::InsufficientData`] if not enough observations.
    /// - [`StatsError::Domain`] if sample moments are out of the parameter space.
    fn fit_mom(data: &[f64]) -> Result<Self>;
}

/// A distribution that supports both MLE and MoM fitting.
///
/// Implemented automatically when both [`MleFit`] and [`MomFit`] are implemented.
pub trait FittableDistribution: MleFit + MomFit {}
impl<T: MleFit + MomFit> FittableDistribution for T {}

// ── Multivariate extension ────────────────────────────────────────────────────

/// A continuous distribution over ℝ^d.
pub trait MultivariateContinuousDistribution {
    /// Dimension d of the distribution.
    fn dim(&self) -> usize;

    /// Log probability density at point `x ∈ ℝ^d`.
    ///
    /// # Errors
    /// [`StatsError::DimensionMismatch`] if `x.len() != self.dim()`.
    fn log_pdf(&self, x: &[f64]) -> Result<f64>;

    /// Probability density at `x`.
    fn pdf(&self, x: &[f64]) -> Result<f64> {
        self.log_pdf(x).map(|lp| lp.exp())
    }

    /// Draw `n` samples, returning an n × d matrix (row-major).
    ///
    /// Each row is one observation.
    fn sample<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        n: usize,
    ) -> Result<crate::types::Matrix>;

    /// Mean vector μ ∈ ℝ^d.
    fn mean_vector(&self) -> crate::types::Vector;

    /// Covariance matrix Σ ∈ ℝ^{d×d}.
    fn covariance_matrix(&self) -> crate::types::Matrix;
}