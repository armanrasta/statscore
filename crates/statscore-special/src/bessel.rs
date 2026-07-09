//! Modified Bessel functions of the first and second kind, orders 0 and 1.
//!
//! These are needed for circular distributions (the von Mises normalizing
//! constant is `I₀`) and various reliability models.
//!
//! # Accuracy
//! - [`i0`], [`i1`] and their scaled forms use a convergent power series for
//!   small/moderate arguments and an asymptotic expansion for large ones,
//!   reaching ~1e-13 relative accuracy.
//! - [`k0`], [`k1`] use the Abramowitz & Stegun polynomial approximations
//!   (§9.8), accurate to ~1e-7. They are candidates for a future Chebyshev
//!   upgrade if tighter tolerances are required.

use std::f64::consts::PI;

/// Above this argument the ascending series for `I₀`/`I₁` risks overflow, so we
/// switch to the (exponentially scaled) asymptotic expansion.
const ASYMP_THRESHOLD: f64 = 700.0;

/// Modified Bessel function of the first kind, order 0: `I₀(x)`.
///
/// `I₀` is even. For `|x| ≳ 709` the true value overflows `f64` and this returns
/// `f64::INFINITY`; use [`i0e`] or [`ln_i0`] for large arguments.
///
/// # Example
/// ```
/// use statscore_special::bessel::i0;
/// assert!((i0(1.0) - 1.266_065_877_752_008_4).abs() < 1e-12);
/// ```
#[must_use]
pub fn i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < ASYMP_THRESHOLD {
        i0_series(ax)
    } else {
        // exp(ax) overflows here; report infinity rather than a wrong finite value.
        f64::INFINITY
    }
}

/// Exponentially scaled `I₀`: returns `e^{-|x|} · I₀(x)`.
///
/// Stays finite and accurate for all `x`, which makes it the right building
/// block for von Mises densities with large concentration.
///
/// # Example
/// ```
/// use statscore_special::bessel::i0e;
/// assert!((i0e(1.0) - 0.465_759_607_593_640_4).abs() < 1e-12);
/// ```
#[must_use]
pub fn i0e(x: f64) -> f64 {
    let ax = x.abs();
    if ax < ASYMP_THRESHOLD {
        i0_series(ax) * (-ax).exp()
    } else {
        i0e_asymptotic(ax)
    }
}

/// Modified Bessel function of the first kind, order 1: `I₁(x)`.
///
/// `I₁` is odd. Returns `±∞` when the magnitude overflows `f64`.
///
/// # Example
/// ```
/// use statscore_special::bessel::i1;
/// assert!((i1(1.0) - 0.565_159_103_992_485).abs() < 1e-12);
/// ```
#[must_use]
pub fn i1(x: f64) -> f64 {
    let ax = x.abs();
    let val = if ax < ASYMP_THRESHOLD {
        i1_series(ax)
    } else {
        f64::INFINITY
    };
    if x < 0.0 { -val } else { val }
}

/// Exponentially scaled `I₁`: returns `e^{-|x|} · I₁(x)` (odd in `x`).
///
/// # Example
/// ```
/// use statscore_special::bessel::{i1, i1e};
/// assert!((i1e(1.0) - i1(1.0) * (-1.0_f64).exp()).abs() < 1e-12);
/// ```
#[must_use]
pub fn i1e(x: f64) -> f64 {
    let ax = x.abs();
    let val = if ax < ASYMP_THRESHOLD {
        i1_series(ax) * (-ax).exp()
    } else {
        i1e_asymptotic(ax)
    };
    if x < 0.0 { -val } else { val }
}

/// Natural logarithm of `I₀(x)`, computed without overflow.
///
/// Equal to `|x| + ln(i0e(x))`, so it is finite and accurate even for very large
/// arguments.
///
/// # Example
/// ```
/// use statscore_special::bessel::{i0, ln_i0};
/// assert!((ln_i0(3.0) - i0(3.0).ln()).abs() < 1e-12);
/// ```
#[must_use]
pub fn ln_i0(x: f64) -> f64 {
    let ax = x.abs();
    ax + i0e(x).ln()
}

/// Modified Bessel function of the second kind, order 0: `K₀(x)`.
///
/// # Domain
/// Defined for `x > 0`. Returns `f64::INFINITY` at `x = 0` and `f64::NAN` for
/// `x < 0`. Accuracy ~1e-7 (Abramowitz & Stegun §9.8).
///
/// # Example
/// ```
/// use statscore_special::bessel::k0;
/// assert!((k0(1.0) - 0.421_024_438_240_708_3).abs() < 1e-6);
/// ```
#[must_use]
pub fn k0(x: f64) -> f64 {
    if x < 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x <= 2.0 {
        let t = (x / 2.0) * (x / 2.0);
        let poly = -0.577_215_66
            + t * (0.422_784_20
                + t * (0.230_697_56
                    + t * (0.034_885_90
                        + t * (0.002_626_98 + t * (0.000_107_50 + t * 0.000_007_40)))));
        -(x / 2.0).ln() * i0_series(x) + poly
    } else {
        k0e(x) * (-x).exp()
    }
}

/// Exponentially scaled `K₀`: returns `e^{x} · K₀(x)`.
///
/// Stays finite for large `x`, where `K₀` itself underflows to zero.
///
/// # Example
/// ```
/// use statscore_special::bessel::k0e;
/// // e^x K0(x) → sqrt(pi/(2x)) as x → ∞
/// let x = 50.0;
/// let asymptote = (std::f64::consts::PI / (2.0 * x)).sqrt();
/// assert!((k0e(x) - asymptote).abs() < 1e-3);
/// ```
#[must_use]
pub fn k0e(x: f64) -> f64 {
    if x < 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x <= 2.0 {
        k0(x) * x.exp()
    } else {
        let z = 2.0 / x;
        let poly = 1.253_314_14
            + z * (-0.078_323_58
                + z * (0.021_895_68
                    + z * (-0.010_624_46
                        + z * (0.005_878_72 + z * (-0.002_515_40 + z * 0.000_532_08)))));
        poly / x.sqrt()
    }
}

/// Modified Bessel function of the second kind, order 1: `K₁(x)`.
///
/// # Domain
/// Defined for `x > 0`. Returns `f64::INFINITY` at `x = 0` and `f64::NAN` for
/// `x < 0`. Accuracy ~1e-7 (Abramowitz & Stegun §9.8).
///
/// # Example
/// ```
/// use statscore_special::bessel::k1;
/// assert!((k1(1.0) - 0.601_907_230_197_234_6).abs() < 1e-6);
/// ```
#[must_use]
pub fn k1(x: f64) -> f64 {
    if x < 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x <= 2.0 {
        let t = (x / 2.0) * (x / 2.0);
        let poly = 1.0
            + t * (0.154_431_44
                + t * (-0.672_785_79
                    + t * (-0.181_568_97
                        + t * (-0.019_194_02 + t * (-0.001_104_04 + t * (-0.000_046_86))))));
        (x / 2.0).ln() * i1_series(x) + poly / x
    } else {
        k1e(x) * (-x).exp()
    }
}

/// Exponentially scaled `K₁`: returns `e^{x} · K₁(x)`.
///
/// # Example
/// ```
/// use statscore_special::bessel::k1e;
/// // e^x K1(x) → sqrt(pi/(2x)) as x → ∞ (slowly; +3/(8x) correction)
/// let x = 50.0;
/// let asymptote = (std::f64::consts::PI / (2.0 * x)).sqrt();
/// assert!((k1e(x) - asymptote).abs() < 2e-3);
/// ```
#[must_use]
pub fn k1e(x: f64) -> f64 {
    if x < 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x <= 2.0 {
        k1(x) * x.exp()
    } else {
        let z = 2.0 / x;
        let poly = 1.253_314_14
            + z * (0.234_986_19
                + z * (-0.036_556_20
                    + z * (0.015_042_68
                        + z * (-0.007_803_53 + z * (0.003_256_14 + z * (-0.000_682_45))))));
        poly / x.sqrt()
    }
}

// ── Internal series / asymptotic kernels ──────────────────────────────────────

/// Ascending power series for `I₀(|x|)`.
fn i0_series(ax: f64) -> f64 {
    let y = (ax * 0.5) * (ax * 0.5);
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1.0;
    loop {
        term *= y / (k * k);
        sum += term;
        if term < sum * 1e-17 || k > 500.0 {
            break;
        }
        k += 1.0;
    }
    sum
}

/// Ascending power series for `I₁(|x|)`.
fn i1_series(ax: f64) -> f64 {
    let y = (ax * 0.5) * (ax * 0.5);
    let mut term = ax * 0.5;
    let mut sum = term;
    let mut k = 1.0;
    loop {
        term *= y / (k * (k + 1.0));
        sum += term;
        if term < sum * 1e-17 || k > 500.0 {
            break;
        }
        k += 1.0;
    }
    sum
}

/// Asymptotic expansion for `e^{-x} I₀(x)`, valid for large `x`.
fn i0e_asymptotic(x: f64) -> f64 {
    let z = 1.0 / x;
    let series = 1.0
        + z * (0.125
            + z * (0.070_312_5
                + z * (0.073_242_187_5
                    + z * (0.112_152_099_609_375 + z * 0.227_108_001_708_984_4))));
    series / (2.0 * PI * x).sqrt()
}

/// Asymptotic expansion for `e^{-x} I₁(x)`, valid for large `x`.
fn i1e_asymptotic(x: f64) -> f64 {
    let z = 1.0 / x;
    let series = 1.0
        + z * (-0.375
            + z * (-0.117_187_5
                + z * (-0.102_539_062_5
                    + z * (-0.144_195_556_640_625 + z * (-0.277_576_446_533_203_1)))));
    series / (2.0 * PI * x).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn i0_reference_values() {
        assert_relative_eq!(i0(0.0), 1.0, epsilon = 1e-15);
        assert_relative_eq!(i0(1.0), 1.266_065_877_752_008_4, max_relative = 1e-13);
        assert_relative_eq!(i0(2.0), 2.279_585_302_336_067_3, max_relative = 1e-13);
        assert_relative_eq!(i0(5.0), 27.239_871_823_604_44, max_relative = 1e-13);
        assert_relative_eq!(i0(-3.0), i0(3.0), epsilon = 1e-15);
    }

    #[test]
    fn i1_reference_values() {
        assert_relative_eq!(i1(0.0), 0.0, epsilon = 1e-15);
        assert_relative_eq!(i1(1.0), 0.565_159_103_992_485, max_relative = 1e-13);
        assert_relative_eq!(i1(2.0), 1.590_636_854_637_329, max_relative = 1e-13);
        assert_relative_eq!(i1(5.0), 24.335_642_142_450_524, max_relative = 1e-13);
        assert_relative_eq!(i1(-2.0), -i1(2.0), epsilon = 1e-15);
    }

    #[test]
    fn i0e_matches_scaled_i0() {
        for &x in &[0.5, 1.0, 5.0, 20.0, 100.0] {
            assert_relative_eq!(i0e(x), i0(x) * (-x).exp(), max_relative = 1e-12);
        }
    }

    #[test]
    fn i0e_large_argument_asymptote() {
        // e^{-x} I0(x) → 1/sqrt(2 pi x)
        let x = 1000.0;
        assert_relative_eq!(i0e(x), 1.0 / (2.0 * PI * x).sqrt(), max_relative = 1e-3);
    }

    #[test]
    fn i1e_matches_scaled_i1() {
        for &x in &[0.5, 1.0, 5.0, 20.0] {
            assert_relative_eq!(i1e(x), i1(x) * (-x).exp(), max_relative = 1e-12);
        }
    }

    #[test]
    fn ln_i0_matches_log_of_i0() {
        for &x in &[0.1, 1.0, 5.0, 50.0] {
            assert_relative_eq!(ln_i0(x), i0(x).ln(), max_relative = 1e-12);
        }
    }

    #[test]
    fn ln_i0_no_overflow_large() {
        // I0(1000) overflows, but ln_i0 stays finite.
        assert!(ln_i0(1000.0).is_finite());
        assert!(i0(1000.0).is_infinite());
    }

    #[test]
    fn k0_reference_values() {
        assert_relative_eq!(k0(0.5), 0.924_419_071_227_666_2, max_relative = 1e-6);
        assert_relative_eq!(k0(1.0), 0.421_024_438_240_708_3, max_relative = 1e-6);
        assert_relative_eq!(k0(2.0), 0.113_893_872_749_533_6, max_relative = 1e-6);
        assert_relative_eq!(k0(5.0), 0.003_691_098_334_042_268, max_relative = 1e-6);
    }

    #[test]
    fn k1_reference_values() {
        assert_relative_eq!(k1(0.5), 1.656_441_120_003_301, max_relative = 1e-6);
        assert_relative_eq!(k1(1.0), 0.601_907_230_197_234_6, max_relative = 1e-6);
        assert_relative_eq!(k1(2.0), 0.139_865_881_816_522_43, max_relative = 1e-6);
        assert_relative_eq!(k1(5.0), 0.004_044_613_445_452_164, max_relative = 1e-6);
    }

    #[test]
    fn k0e_matches_scaled_k0() {
        for &x in &[0.5, 1.0, 2.0, 5.0] {
            assert_relative_eq!(k0e(x), k0(x) * x.exp(), max_relative = 1e-6);
        }
    }

    #[test]
    fn k1e_matches_scaled_k1() {
        for &x in &[0.5, 1.0, 2.0, 5.0] {
            assert_relative_eq!(k1e(x), k1(x) * x.exp(), max_relative = 1e-6);
        }
    }

    #[test]
    fn bessel_k_domain() {
        assert!(k0(-1.0).is_nan());
        assert!(k1(-1.0).is_nan());
        assert!(k0(0.0).is_infinite());
        assert!(k1(0.0).is_infinite());
    }
}
