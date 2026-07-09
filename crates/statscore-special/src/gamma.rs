//! The gamma function and its relatives.
//!
//! This module provides:
//! - [`ln_gamma`] — natural log of the absolute value of Γ(x)
//! - [`gamma`] — the gamma function Γ(x)
//! - [`digamma`] — ψ(x) = d/dx ln Γ(x)
//! - [`trigamma`] — ψ₁(x) = d²/dx² ln Γ(x)
//! - [`gammainc`] / [`gammaincc`] — the regularized lower / upper incomplete
//!   gamma functions P(a, x) and Q(a, x)
//!
//! `ln_gamma` and `gamma` use the Lanczos approximation (g = 7, 9 coefficients),
//! which is accurate to roughly 1e-15 relative error across the real line.

use std::f64::consts::PI;

/// The Euler–Mascheroni constant γ ≈ 0.5772156649.
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// ln(√(2π)), used by the Lanczos formula.
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

/// The Lanczos parameter `g`.
const LANCZOS_G: f64 = 7.0;

/// Lanczos coefficients for `g = 7`, `n = 9` (Godfrey's values).
const LANCZOS_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Natural logarithm of the absolute value of the gamma function, `ln|Γ(x)|`.
///
/// Uses the Lanczos approximation together with the reflection formula for
/// `x < 0.5`. Accurate to ~1e-15 relative error.
///
/// # Special values
/// - Returns `f64::INFINITY` at the poles `x = 0, -1, -2, …`.
/// - `ln_gamma(1) == 0` and `ln_gamma(2) == 0`.
///
/// # Example
/// ```
/// use statscore_special::gamma::ln_gamma;
/// // Γ(5) = 4! = 24
/// assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-12);
/// ```
#[must_use]
pub fn ln_gamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Poles at non-positive integers.
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
    }
    // Exact zeros: Γ(1) = Γ(2) = 1. Pin these so ln_factorial(0)/(1) are exact.
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }

    if x < 0.5 {
        // Reflection: Γ(x)Γ(1−x) = π / sin(πx)
        // ln|Γ(x)| = ln(π) − ln|sin(πx)| − ln|Γ(1−x)|
        let sin_pix = (PI * x).sin().abs();
        PI.ln() - sin_pix.ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = LANCZOS_COEFFS[0];
        let t = x + LANCZOS_G + 0.5;
        for (i, &c) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        LN_SQRT_2PI + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// The gamma function Γ(x).
///
/// Uses the Lanczos approximation directly (so the sign is correct for negative
/// non-integer arguments) with the reflection formula for `x < 0.5`.
///
/// # Special values
/// - Returns `f64::NAN` at the poles `x = 0, -1, -2, …`.
/// - `gamma(n) == (n-1)!` for positive integers.
///
/// # Example
/// ```
/// use statscore_special::gamma::gamma;
/// assert!((gamma(6.0) - 120.0).abs() < 1e-9);   // 5! = 120
/// assert!((gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-12);
/// ```
#[must_use]
pub fn gamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 && x == x.floor() {
        return f64::NAN;
    }

    if x < 0.5 {
        // Γ(x) = π / (sin(πx) · Γ(1−x))
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut a = LANCZOS_COEFFS[0];
        let t = x + LANCZOS_G + 0.5;
        for (i, &c) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        let two_pi = 2.0 * PI;
        two_pi.sqrt() * t.powf(x + 0.5) * (-t).exp() * a
    }
}

/// The digamma function ψ(x) = Γ′(x) / Γ(x).
///
/// Combines the reflection formula (for `x ≤ 0`), upward recurrence, and the
/// standard asymptotic expansion. Accurate to ~1e-13.
///
/// # Special values
/// Returns `f64::NAN` at the poles `x = 0, -1, -2, …`.
///
/// # Example
/// ```
/// use statscore_special::gamma::digamma;
/// // ψ(1) = −γ (negative Euler–Mascheroni constant)
/// assert!((digamma(1.0) + 0.577_215_664_901_532_9).abs() < 1e-12);
/// ```
#[must_use]
pub fn digamma(mut x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 && x == x.floor() {
        return f64::NAN;
    }

    let mut result = 0.0;

    // Reflection for non-positive arguments: ψ(1−x) − ψ(x) = π cot(πx)
    if x < 0.0 {
        result -= PI / (PI * x).tan();
        x = 1.0 - x;
    }

    // Upward recurrence until x is large enough for the asymptotic series.
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion:
    // ψ(x) ≈ ln x − 1/(2x) − Σ B_{2k} / (2k x^{2k})
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result += x.ln() - 0.5 * inv;
    result -= inv2
        * (1.0 / 12.0
            - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 / 132.0))));
    result
}

/// The trigamma function ψ₁(x) = d/dx ψ(x).
///
/// Uses the reflection formula, upward recurrence, and asymptotic expansion.
///
/// # Special values
/// Returns `f64::INFINITY` at the poles `x = 0, -1, -2, …`.
///
/// # Example
/// ```
/// use statscore_special::gamma::trigamma;
/// // ψ₁(1) = π²/6
/// let expected = std::f64::consts::PI.powi(2) / 6.0;
/// assert!((trigamma(1.0) - expected).abs() < 1e-10);
/// ```
#[must_use]
pub fn trigamma(mut x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
    }

    let mut result = 0.0;

    // Reflection: ψ₁(1−x) + ψ₁(x) = π² / sin²(πx)
    if x < 0.0 {
        let s = (PI * x).sin();
        return PI * PI / (s * s) - trigamma(1.0 - x);
    }

    // Upward recurrence: ψ₁(x) = ψ₁(x+1) + 1/x²
    while x < 10.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }

    // Asymptotic: ψ₁(x) ≈ 1/x + 1/(2x²) + Σ B_{2k}/x^{2k+1}
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result += inv
        * (1.0
            + inv
                * (0.5
                    + inv * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 * (1.0 / 42.0 - inv2 / 30.0)))));
    result
}

/// Maximum iterations for the incomplete-gamma series / continued fraction.
const ITMAX: usize = 300;
/// Relative convergence tolerance for incomplete-gamma iterations.
const EPS: f64 = 1e-15;
/// A tiny number to guard against division by zero in the continued fraction.
const FPMIN: f64 = 1e-300;

/// The regularized lower incomplete gamma function P(a, x) = γ(a, x) / Γ(a).
///
/// This is the CDF of a Gamma(shape = `a`, scale = 1) distribution.
///
/// # Special values
/// - `P(a, 0) = 0`, `P(a, ∞) = 1`.
/// - Returns `f64::NAN` if `a ≤ 0` or `x < 0`.
///
/// # Example
/// ```
/// use statscore_special::gamma::gammainc;
/// // P(1, x) = 1 − e^{−x}  (exponential CDF)
/// assert!((gammainc(1.0, 2.0) - (1.0 - (-2.0_f64).exp())).abs() < 1e-12);
/// ```
#[must_use]
pub fn gammainc(a: f64, x: f64) -> f64 {
    if a <= 0.0 || x < 0.0 || a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cf(a, x)
    }
}

/// The regularized upper incomplete gamma function Q(a, x) = Γ(a, x) / Γ(a).
///
/// Equal to `1 − gammainc(a, x)` but computed directly in the regime where that
/// is the more accurate branch, avoiding cancellation.
///
/// # Special values
/// - `Q(a, 0) = 1`, `Q(a, ∞) = 0`.
/// - Returns `f64::NAN` if `a ≤ 0` or `x < 0`.
///
/// # Example
/// ```
/// use statscore_special::gamma::{gammainc, gammaincc};
/// let (a, x) = (3.5, 2.0);
/// assert!((gammainc(a, x) + gammaincc(a, x) - 1.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn gammaincc(a: f64, x: f64) -> f64 {
    if a <= 0.0 || x < 0.0 || a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_series(a, x)
    } else {
        gamma_cf(a, x)
    }
}

/// Series representation of P(a, x), valid for `x < a + 1`.
fn gamma_series(a: f64, x: f64) -> f64 {
    let gln = ln_gamma(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..ITMAX {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            break;
        }
    }
    sum * (-x + a * x.ln() - gln).exp()
}

/// Continued-fraction representation of Q(a, x), valid for `x ≥ a + 1`.
fn gamma_cf(a: f64, x: f64) -> f64 {
    let gln = ln_gamma(a);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=ITMAX {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    (-x + a * x.ln() - gln).exp() * h
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn gamma_integers() {
        assert_relative_eq!(gamma(1.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(gamma(2.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(gamma(6.0), 120.0, epsilon = 1e-9);
        assert_relative_eq!(gamma(11.0), 3_628_800.0, max_relative = 1e-10);
    }

    #[test]
    fn gamma_half_integers() {
        let sqrt_pi = PI.sqrt();
        assert_relative_eq!(gamma(0.5), sqrt_pi, epsilon = 1e-12);
        assert_relative_eq!(gamma(1.5), sqrt_pi / 2.0, epsilon = 1e-12);
        assert_relative_eq!(gamma(2.5), 3.0 * sqrt_pi / 4.0, epsilon = 1e-12);
    }

    #[test]
    fn gamma_negative_noninteger() {
        // Γ(-0.5) = -2√π
        assert_relative_eq!(gamma(-0.5), -2.0 * PI.sqrt(), epsilon = 1e-11);
    }

    #[test]
    fn gamma_poles_are_nan() {
        assert!(gamma(0.0).is_nan());
        assert!(gamma(-1.0).is_nan());
        assert!(gamma(-5.0).is_nan());
    }

    #[test]
    fn ln_gamma_matches_gamma_ln() {
        for &x in &[0.1, 0.7, 1.3, 2.5, 7.9, 20.0, 50.0] {
            assert_relative_eq!(ln_gamma(x), gamma(x).ln(), max_relative = 1e-10);
        }
    }

    #[test]
    fn ln_gamma_reference_values() {
        // ln Γ(100) = 359.1342053695754 (SciPy gammaln)
        assert_relative_eq!(ln_gamma(100.0), 359.134_205_369_575_4, max_relative = 1e-13);
        // ln Γ(0.1) = 2.252712651734206
        assert_relative_eq!(ln_gamma(0.1), 2.252_712_651_734_206, max_relative = 1e-13);
    }

    #[test]
    fn digamma_reference_values() {
        assert_relative_eq!(digamma(1.0), -EULER_GAMMA, epsilon = 1e-12);
        // ψ(0.5) = −γ − 2 ln 2
        assert_relative_eq!(
            digamma(0.5),
            -EULER_GAMMA - 2.0 * 2.0_f64.ln(),
            epsilon = 1e-12
        );
        // ψ(10) = 2.251752589066721 (SciPy)
        assert_relative_eq!(digamma(10.0), 2.251_752_589_066_721, max_relative = 1e-12);
    }

    #[test]
    fn trigamma_reference_values() {
        assert_relative_eq!(trigamma(1.0), PI * PI / 6.0, max_relative = 1e-11);
        // ψ₁(0.5) = π²/2
        assert_relative_eq!(trigamma(0.5), PI * PI / 2.0, max_relative = 1e-11);
        // ψ₁(10) = 0.10516633568168575 (SciPy)
        assert_relative_eq!(
            trigamma(10.0),
            0.105_166_335_681_685_75,
            max_relative = 1e-10
        );
    }

    #[test]
    fn gammainc_exponential_cdf() {
        // P(1, x) = 1 − e^{−x}
        for &x in &[0.1, 1.0, 3.0, 10.0] {
            assert_relative_eq!(gammainc(1.0, x), 1.0 - (-x).exp(), epsilon = 1e-12);
        }
    }

    #[test]
    fn gammainc_complement_sums_to_one() {
        for &(a, x) in &[(0.5, 0.3), (2.0, 1.0), (5.0, 7.0), (10.0, 3.0), (3.5, 12.0)] {
            assert_relative_eq!(gammainc(a, x) + gammaincc(a, x), 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn gammainc_reference_values() {
        // scipy.special.gammainc(3, 5) = 0.8753479805169189
        assert_relative_eq!(
            gammainc(3.0, 5.0),
            0.875_347_980_516_918_9,
            max_relative = 1e-11
        );
        // scipy.special.gammainc(0.5, 1) = 0.8427007929497149 (= erf(1))
        assert_relative_eq!(
            gammainc(0.5, 1.0),
            0.842_700_792_949_714_9,
            max_relative = 1e-11
        );
    }

    #[test]
    fn gammainc_domain_errors() {
        assert!(gammainc(-1.0, 1.0).is_nan());
        assert!(gammainc(1.0, -1.0).is_nan());
        assert_eq!(gammainc(2.0, 0.0), 0.0);
        assert_eq!(gammaincc(2.0, 0.0), 1.0);
    }
}
