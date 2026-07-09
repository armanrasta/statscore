//! Log-space combinatorial functions.
//!
//! These underpin the PMFs of discrete distributions (binomial, Poisson,
//! hypergeometric, …). Everything is computed in log space through
//! [`ln_gamma`](crate::gamma::ln_gamma) so it stays finite for large arguments.

use crate::gamma::ln_gamma;

/// Largest `n` for which `n!` is representable exactly as an `f64`.
///
/// `170! ≈ 7.257e306` fits; `171!` overflows.
const MAX_EXACT_FACTORIAL: u64 = 170;

/// Natural logarithm of `n!`, i.e. `ln Γ(n + 1)`.
///
/// # Example
/// ```
/// use statscore_special::combinatorics::ln_factorial;
/// // ln(5!) = ln(120)
/// assert!((ln_factorial(5) - 120.0_f64.ln()).abs() < 1e-12);
/// assert_eq!(ln_factorial(0), 0.0);
/// ```
#[must_use]
pub fn ln_factorial(n: u64) -> f64 {
    ln_gamma(n as f64 + 1.0)
}

/// The factorial `n!` as an `f64`.
///
/// Exact (up to `f64` rounding) for `n ≤ 170`; returns `f64::INFINITY` beyond
/// that, where the true value exceeds `f64::MAX`.
///
/// # Example
/// ```
/// use statscore_special::combinatorics::factorial;
/// assert_eq!(factorial(5), 120.0);
/// assert!(factorial(171).is_infinite());
/// ```
#[must_use]
pub fn factorial(n: u64) -> f64 {
    if n > MAX_EXACT_FACTORIAL {
        return f64::INFINITY;
    }
    let mut acc = 1.0_f64;
    for k in 2..=n {
        acc *= k as f64;
    }
    acc
}

/// Natural logarithm of the binomial coefficient `ln C(n, k)`.
///
/// Returns `f64::NEG_INFINITY` when `k > n` (the coefficient is zero).
///
/// # Example
/// ```
/// use statscore_special::combinatorics::ln_choose;
/// // C(10, 3) = 120
/// assert!((ln_choose(10, 3) - 120.0_f64.ln()).abs() < 1e-12);
/// assert_eq!(ln_choose(3, 5), f64::NEG_INFINITY);
/// ```
#[must_use]
pub fn ln_choose(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
}

/// The binomial coefficient `C(n, k)` as an `f64`.
///
/// Returns `0.0` when `k > n`. Computed via [`ln_choose`] and exponentiated, then
/// rounded to the nearest integer for the range where that is exact.
///
/// # Example
/// ```
/// use statscore_special::combinatorics::choose;
/// assert_eq!(choose(10, 3), 120.0);
/// assert_eq!(choose(3, 5), 0.0);
/// ```
#[must_use]
pub fn choose(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    let v = ln_choose(n, k).exp();
    // For values that fit comfortably in an f64 mantissa the result is an
    // integer; round away the tiny exp/ln error.
    if v < 9.007_199_254_740_992e15 {
        v.round()
    } else {
        v
    }
}

/// Natural logarithm of the number of permutations `ln P(n, k) = ln(n! / (n−k)!)`.
///
/// Returns `f64::NEG_INFINITY` when `k > n`.
///
/// # Example
/// ```
/// use statscore_special::combinatorics::ln_perm;
/// // P(5, 2) = 20
/// assert!((ln_perm(5, 2) - 20.0_f64.ln()).abs() < 1e-12);
/// ```
#[must_use]
pub fn ln_perm(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_factorial(n) - ln_factorial(n - k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn factorial_small() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(1), 1.0);
        assert_eq!(factorial(5), 120.0);
        assert_eq!(factorial(10), 3_628_800.0);
    }

    #[test]
    fn factorial_overflow() {
        assert!(factorial(170).is_finite());
        assert!(factorial(171).is_infinite());
    }

    #[test]
    fn ln_factorial_reference() {
        assert_eq!(ln_factorial(0), 0.0);
        assert_relative_eq!(ln_factorial(10), 3_628_800.0_f64.ln(), max_relative = 1e-13);
        // ln(100!) = 363.7393755555635 (SciPy gammaln(101))
        assert_relative_eq!(ln_factorial(100), 363.739_375_555_563_5, max_relative = 1e-13);
    }

    #[test]
    fn choose_values() {
        assert_eq!(choose(10, 0), 1.0);
        assert_eq!(choose(10, 10), 1.0);
        assert_eq!(choose(10, 3), 120.0);
        assert_eq!(choose(52, 5), 2_598_960.0);
        assert_eq!(choose(3, 5), 0.0);
    }

    #[test]
    fn ln_choose_matches_choose() {
        for &(n, k) in &[(10, 3), (20, 7), (50, 25), (100, 40)] {
            assert_relative_eq!(ln_choose(n, k).exp(), choose(n, k), max_relative = 1e-9);
        }
    }

    #[test]
    fn ln_choose_out_of_range() {
        assert_eq!(ln_choose(3, 5), f64::NEG_INFINITY);
    }

    #[test]
    fn ln_perm_values() {
        assert_relative_eq!(ln_perm(5, 2).exp(), 20.0, max_relative = 1e-10);
        assert_relative_eq!(ln_perm(10, 3).exp(), 720.0, max_relative = 1e-10);
        assert_eq!(ln_perm(3, 5), f64::NEG_INFINITY);
    }
}
