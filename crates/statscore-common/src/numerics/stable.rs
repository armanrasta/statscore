//! Numerically stable elementary operations.
//!
//! These prevent overflow/underflow in log-space computations that appear
//! throughout distribution log-likelihoods, MCMC samplers, and softmax layers.

/// Numerically stable log(Σ exp(xᵢ)).
///
/// Uses the identity:
/// log Σ exp(xᵢ) = max(x) + log Σ exp(xᵢ − max(x))
///
/// Returns `f64::NEG_INFINITY` for an empty slice.
///
/// # Example
/// ```
/// use statscore_common::numerics::stable::log_sum_exp;
/// let vals = [1.0_f64.ln(), 2.0_f64.ln(), 3.0_f64.ln()];
/// let result = log_sum_exp(&vals);
/// assert!((result - 6.0_f64.ln()).abs() < 1e-12);
/// ```
pub fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // If max is -inf, all values are -inf → result is -inf.
    // If max is +inf, result is +inf.
    if !max.is_finite() {
        return max;
    }

    let sum: f64 = values.iter().map(|v| (v - max).exp()).sum();
    max + sum.ln()
}

/// Numerically stable softmax: exp(xᵢ) / Σ exp(xⱼ).
///
/// Returns a vector of the same length that sums to 1.0.
///
/// # Panics
/// Never panics; returns an empty `Vec` for empty input.
///
/// # Example
/// ```
/// use statscore_common::numerics::stable::softmax;
/// let sm = softmax(&[1.0, 2.0, 3.0]);
/// let total: f64 = sm.iter().sum();
/// assert!((total - 1.0).abs() < 1e-12);
/// ```
pub fn softmax(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let lse = log_sum_exp(values);
    values.iter().map(|v| (v - lse).exp()).collect()
}

/// log(1 + exp(x)) — the softplus function — computed without overflow.
///
/// Branches:
/// - x ≤ −37   → exp(x)         (exp(x) ≈ 0, so log(1+exp(x)) ≈ exp(x))
/// - x ≤  18   → log(1 + exp(x)) (direct, safe)
/// - x ≤  33.3 → x + exp(−x)   (exp(x) large, log(1+exp(x)) ≈ x + exp(−x))
/// - x  >  33.3 → x             (exp(−x) negligible)
///
/// # Example
/// ```
/// use statscore_common::numerics::stable::log1pexp;
/// assert!((log1pexp(0.0) - 2.0_f64.ln()).abs() < 1e-15);
/// assert!((log1pexp(100.0) - 100.0).abs() < 1e-10);
/// ```
pub fn log1pexp(x: f64) -> f64 {
    if x <= -37.0 {
        x.exp()
    } else if x <= 18.0 {
        x.exp().ln_1p()
    } else if x <= 33.3 {
        x + (-x).exp()
    } else {
        x
    }
}

/// log(1 − exp(x)) for x < 0, computed stably.
///
/// Used in CDFs and log-probabilities where the argument is a negative log-prob.
///
/// # Domain
/// `x` must be strictly negative. Returns `f64::NAN` for `x >= 0`.
///
/// # Example
/// ```
/// use statscore_common::numerics::stable::log1mexp;
/// // log(1 - exp(-1)) ≈ log(0.6321) ≈ -0.4587
/// let result = log1mexp(-1.0);
/// assert!((result - (1.0_f64 - (-1.0_f64).exp()).ln()).abs() < 1e-14);
/// ```
pub fn log1mexp(x: f64) -> f64 {
    if x >= 0.0 {
        return f64::NAN;
    }
    // For x in (-ln2, 0): use log(-expm1(x))
    // For x <= -ln2:      use log1p(-exp(x))
    if x > -std::f64::consts::LN_2 {
        (-x.exp_m1()).ln()
    } else {
        (-x).exp().ln_1p()
    }
}

/// log(exp(a) + exp(b)) computed without overflow.
///
/// Equivalent to `log_sum_exp(&[a, b])` but avoids the slice allocation.
///
/// # Example
/// ```
/// use statscore_common::numerics::stable::log_add;
/// let result = log_add(1.0_f64.ln(), 2.0_f64.ln());
/// assert!((result - 3.0_f64.ln()).abs() < 1e-12);
/// ```
pub fn log_add(a: f64, b: f64) -> f64 {
    // Handle infinities explicitly to avoid NaN from inf - inf.
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    if a >= b {
        a + log1pexp(b - a)
    } else {
        b + log1pexp(a - b)
    }
}

/// Logistic (sigmoid) function: 1 / (1 + exp(−x)).
///
/// Numerically stable for large positive and large negative x.
///
/// # Example
/// ```
/// use statscore_common::numerics::stable::logistic;
/// assert!((logistic(0.0) - 0.5).abs() < 1e-15);
/// assert!((logistic(100.0) - 1.0).abs() < 1e-12);
/// ```
pub fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// log-logistic: log(1 / (1 + exp(−x))) = −log1pexp(−x).
///
/// This is the log of the sigmoid, used in logistic regression log-likelihoods.
pub fn log_logistic(x: f64) -> f64 {
    -log1pexp(-x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn log_sum_exp_basic() {
        let vals = [1.0_f64.ln(), 2.0_f64.ln(), 3.0_f64.ln()];
        assert_relative_eq!(log_sum_exp(&vals), 6.0_f64.ln(), epsilon = 1e-12);
    }

    #[test]
    fn log_sum_exp_single() {
        assert_relative_eq!(log_sum_exp(&[3.7]), 3.7, epsilon = 1e-15);
    }

    #[test]
    fn log_sum_exp_empty() {
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn log_sum_exp_large_values_no_overflow() {
        let vals = [1000.0, 1001.0, 1002.0];
        let result = log_sum_exp(&vals);
        assert!(result.is_finite());
        // Should be close to 1002 + log(1 + e^-1 + e^-2)
        assert!(result > 1002.0 && result < 1003.0);
    }

    #[test]
    fn softmax_sums_to_one() {
        let sm = softmax(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_relative_eq!(sm.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn softmax_all_equal() {
        let sm = softmax(&[2.0, 2.0, 2.0, 2.0]);
        for v in &sm {
            assert_relative_eq!(*v, 0.25, epsilon = 1e-14);
        }
    }

    #[test]
    fn softmax_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn log1pexp_zero() {
        // log(1 + exp(0)) = log(2)
        assert_relative_eq!(log1pexp(0.0), 2.0_f64.ln(), epsilon = 1e-15);
    }

    #[test]
    fn log1pexp_large_positive() {
        assert_relative_eq!(log1pexp(100.0), 100.0, epsilon = 1e-10);
    }

    #[test]
    fn log1pexp_large_negative() {
        // Should be essentially 0
        assert!(log1pexp(-100.0) < 1e-40);
    }

    #[test]
    fn log1mexp_correctness() {
        // log(1 - exp(-1)) direct
        let expected = (1.0 - (-1.0_f64).exp()).ln();
        assert_relative_eq!(log1mexp(-1.0), expected, epsilon = 1e-14);
    }

    #[test]
    fn log1mexp_domain_violation() {
        assert!(log1mexp(0.0).is_nan());
        assert!(log1mexp(1.0).is_nan());
    }

    #[test]
    fn log_add_basic() {
        let result = log_add(1.0_f64.ln(), 2.0_f64.ln());
        assert_relative_eq!(result, 3.0_f64.ln(), epsilon = 1e-12);
    }

    #[test]
    fn log_add_neg_infinity() {
        assert_eq!(log_add(f64::NEG_INFINITY, 5.0), 5.0);
        assert_eq!(log_add(5.0, f64::NEG_INFINITY), 5.0);
    }

    #[test]
    fn logistic_at_zero() {
        assert_relative_eq!(logistic(0.0), 0.5, epsilon = 1e-15);
    }

    #[test]
    fn logistic_large_positive() {
        assert_relative_eq!(logistic(100.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn logistic_large_negative() {
        assert_relative_eq!(logistic(-100.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn log_logistic_consistency() {
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0] {
            assert_relative_eq!(
                log_logistic(x),
                logistic(x).ln(),
                epsilon = 1e-12
            );
        }
    }
}