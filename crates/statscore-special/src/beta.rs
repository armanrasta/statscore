//! The beta function and the regularized incomplete beta function.
//!
//! - [`ln_beta`] / [`beta`] — B(a, b) = Γ(a)Γ(b)/Γ(a+b)
//! - [`betainc`] — the regularized incomplete beta I_x(a, b), i.e. the CDF of a
//!   Beta(a, b) distribution
//! - [`betaincinv`] — its inverse in `x`

use crate::gamma::ln_gamma;

/// Natural logarithm of the beta function, `ln B(a, b)`.
///
/// # Special values
/// Returns `f64::NAN` if `a ≤ 0` or `b ≤ 0`.
///
/// # Example
/// ```
/// use statscore_special::beta::ln_beta;
/// // B(2, 3) = 1/12
/// assert!((ln_beta(2.0, 3.0) - (1.0_f64 / 12.0).ln()).abs() < 1e-12);
/// ```
#[must_use]
pub fn ln_beta(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// The beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
///
/// # Special values
/// Returns `f64::NAN` if `a ≤ 0` or `b ≤ 0`.
///
/// # Example
/// ```
/// use statscore_special::beta::beta;
/// assert!((beta(2.0, 3.0) - 1.0 / 12.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn beta(a: f64, b: f64) -> f64 {
    ln_beta(a, b).exp()
}

/// Maximum iterations for the incomplete-beta continued fraction.
const ITMAX: usize = 300;
/// Relative convergence tolerance.
const EPS: f64 = 1e-15;
/// Guard against division by zero in the continued fraction.
const FPMIN: f64 = 1e-300;

/// The regularized incomplete beta function I_x(a, b).
///
/// This is the CDF of a Beta(a, b) distribution evaluated at `x`.
///
/// # Special values
/// - `I_0(a, b) = 0`, `I_1(a, b) = 1`.
/// - Returns `f64::NAN` if `a ≤ 0`, `b ≤ 0`, or `x ∉ [0, 1]`.
///
/// # Example
/// ```
/// use statscore_special::beta::betainc;
/// // Symmetry: I_x(a, b) = 1 − I_{1−x}(b, a)
/// let (a, b, x) = (2.0, 3.0, 0.4);
/// assert!((betainc(a, b, x) - (1.0 - betainc(b, a, 1.0 - x))).abs() < 1e-12);
/// ```
#[must_use]
pub fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || x < 0.0 || x > 1.0 || a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 || x == 1.0 {
        return x;
    }

    // Factor in front of the continued fraction.
    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();

    // Use whichever argument converges faster.
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x) / a
    } else {
        1.0 - front * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Lentz's continued fraction for the incomplete beta function.
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=ITMAX {
        let m = m as f64;
        let m2 = 2.0 * m;

        // Even step.
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step.
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
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
    h
}

/// The inverse of the regularized incomplete beta function in `x`.
///
/// Returns the value `x ∈ [0, 1]` such that `betainc(a, b, x) = p`; this is the
/// quantile function (PPF) of a Beta(a, b) distribution.
///
/// # Special values
/// Returns `f64::NAN` if `a ≤ 0`, `b ≤ 0`, or `p ∉ [0, 1]`.
///
/// # Example
/// ```
/// use statscore_special::beta::{betainc, betaincinv};
/// let (a, b, p) = (2.5, 4.0, 0.42);
/// let x = betaincinv(a, b, p);
/// assert!((betainc(a, b, x) - p).abs() < 1e-10);
/// ```
#[must_use]
pub fn betaincinv(a: f64, b: f64, p: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || p < 0.0 || p > 1.0 || a.is_nan() || b.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if p == 0.0 || p == 1.0 {
        return p;
    }

    // Bisection is robust and I_x is monotone increasing in x; refine with a few
    // Newton steps using the Beta pdf as the derivative.
    let ln_beta_ab = ln_beta(a, b);
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut x = 0.5;

    for _ in 0..200 {
        let err = betainc(a, b, x) - p;
        if err.abs() < 1e-14 {
            break;
        }
        if err > 0.0 {
            hi = x;
        } else {
            lo = x;
        }

        // Newton candidate: x - f(x)/f'(x), f'(x) = pdf of Beta(a,b).
        let ln_pdf = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - ln_beta_ab;
        let deriv = ln_pdf.exp();
        let newton = if deriv > 0.0 { x - err / deriv } else { f64::NAN };

        x = if newton.is_finite() && newton > lo && newton < hi {
            newton
        } else {
            0.5 * (lo + hi)
        };
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn beta_reference_values() {
        assert_relative_eq!(beta(2.0, 3.0), 1.0 / 12.0, max_relative = 1e-12);
        assert_relative_eq!(beta(1.0, 1.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(beta(0.5, 0.5), std::f64::consts::PI, max_relative = 1e-12);
    }

    #[test]
    fn ln_beta_symmetry() {
        for &(a, b) in &[(2.0, 5.0), (0.3, 1.7), (10.0, 4.0)] {
            assert_relative_eq!(ln_beta(a, b), ln_beta(b, a), max_relative = 1e-13);
        }
    }

    #[test]
    fn betainc_boundaries() {
        assert_eq!(betainc(2.0, 3.0, 0.0), 0.0);
        assert_eq!(betainc(2.0, 3.0, 1.0), 1.0);
    }

    #[test]
    fn betainc_symmetry() {
        for &(a, b, x) in &[(2.0, 3.0, 0.4), (0.5, 2.5, 0.8), (5.0, 1.0, 0.2)] {
            assert_relative_eq!(
                betainc(a, b, x),
                1.0 - betainc(b, a, 1.0 - x),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn betainc_reference_values() {
        // scipy.special.betainc(2, 3, 0.5) = 0.6875
        assert_relative_eq!(betainc(2.0, 3.0, 0.5), 0.6875, max_relative = 1e-12);
        // scipy.special.betainc(0.5, 0.5, 0.5) = 0.5
        assert_relative_eq!(betainc(0.5, 0.5, 0.5), 0.5, epsilon = 1e-12);
        // scipy.special.betainc(2, 5, 0.3) = 0.7443100000000001
        assert_relative_eq!(betainc(2.0, 5.0, 0.3), 0.744_31, max_relative = 1e-11);
    }

    #[test]
    fn betaincinv_roundtrips() {
        for &(a, b, p) in &[
            (2.0, 3.0, 0.25),
            (0.5, 0.5, 0.5),
            (5.0, 2.0, 0.9),
            (2.5, 4.0, 0.42),
            (1.0, 1.0, 0.7),
        ] {
            let x = betaincinv(a, b, p);
            assert_relative_eq!(betainc(a, b, x), p, epsilon = 1e-10);
        }
    }

    #[test]
    fn domain_errors() {
        assert!(beta(-1.0, 2.0).is_nan());
        assert!(betainc(2.0, 3.0, 1.5).is_nan());
        assert!(betaincinv(2.0, 3.0, -0.1).is_nan());
    }
}
