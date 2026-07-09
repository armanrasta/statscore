//! The error function and its inverse.
//!
//! - [`erf`] / [`erfc`] delegate to `libm`, which provides correctly-rounded
//!   double-precision implementations that compile on every target.
//! - [`erf_inv`] / [`erfc_inv`] use a rational first guess refined by two
//!   Halley steps, reaching full double precision.

/// 2/√π, the derivative constant d/dx erf(x) = (2/√π) e^{−x²}.
const TWO_OVER_SQRT_PI: f64 = 1.128_379_167_095_512_6;

/// The error function erf(x) = (2/√π) ∫₀ˣ e^{−t²} dt.
///
/// # Example
/// ```
/// use statscore_special::erf::erf;
/// assert!((erf(0.0)).abs() < 1e-15);
/// assert!((erf(f64::INFINITY) - 1.0).abs() < 1e-15);
/// ```
#[must_use]
pub fn erf(x: f64) -> f64 {
    libm::erf(x)
}

/// The complementary error function erfc(x) = 1 − erf(x).
///
/// Computed directly (not as `1 - erf(x)`) so it stays accurate in the tail
/// where `erf(x) → 1`.
///
/// # Example
/// ```
/// use statscore_special::erf::erfc;
/// assert!((erfc(0.0) - 1.0).abs() < 1e-15);
/// ```
#[must_use]
pub fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

/// The inverse error function: returns `x` such that `erf(x) = y`.
///
/// # Domain
/// `y` must lie in `[-1, 1]`. Returns `±∞` at `±1` and `f64::NAN` outside.
///
/// # Example
/// ```
/// use statscore_special::erf::{erf, erf_inv};
/// let y = 0.75;
/// assert!((erf(erf_inv(y)) - y).abs() < 1e-12);
/// ```
#[must_use]
pub fn erf_inv(y: f64) -> f64 {
    erfc_inv(1.0 - y)
}

/// The inverse complementary error function: returns `x` such that
/// `erfc(x) = p`.
///
/// # Domain
/// `p` must lie in `[0, 2]`. Returns `+∞` at `0`, `−∞` at `2`, and `f64::NAN`
/// outside.
///
/// # Example
/// ```
/// use statscore_special::erf::{erfc, erfc_inv};
/// let p = 0.3;
/// assert!((erfc(erfc_inv(p)) - p).abs() < 1e-12);
/// ```
#[must_use]
pub fn erfc_inv(p: f64) -> f64 {
    if p.is_nan() || p < 0.0 || p > 2.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p == 2.0 {
        return f64::NEG_INFINITY;
    }

    // Fold to the lower half and remember the sign.
    let pp = if p < 1.0 { p } else { 2.0 - p };
    let t = (-2.0 * (pp / 2.0).ln()).sqrt();

    // Rational initial guess (Numerical Recipes, 3rd ed.).
    let mut x = -std::f64::consts::FRAC_1_SQRT_2
        * ((2.307_53 + t * 0.270_61) / (1.0 + t * (0.992_29 + t * 0.044_81)) - t);

    // Two Halley iterations refine to full precision.
    for _ in 0..2 {
        let err = erfc(x) - pp;
        x += err / (TWO_OVER_SQRT_PI * (-x * x).exp() - x * err);
    }

    if p < 1.0 { x } else { -x }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn erf_reference_values() {
        assert_relative_eq!(erf(0.5), 0.520_499_877_813_046_5, max_relative = 1e-14);
        assert_relative_eq!(erf(1.0), 0.842_700_792_949_714_9, max_relative = 1e-14);
        assert_relative_eq!(erf(2.0), 0.995_322_265_018_952_7, max_relative = 1e-14);
        assert_relative_eq!(erf(-1.0), -0.842_700_792_949_714_9, max_relative = 1e-14);
    }

    #[test]
    fn erfc_complements_erf() {
        for &x in &[-3.0, -1.0, 0.0, 0.5, 2.0, 4.0] {
            assert_relative_eq!(erf(x) + erfc(x), 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn erf_inv_roundtrips() {
        for &y in &[-0.99, -0.5, -0.01, 0.0, 0.3, 0.75, 0.999] {
            assert_relative_eq!(erf(erf_inv(y)), y, epsilon = 1e-12);
        }
    }

    #[test]
    fn erfc_inv_roundtrips() {
        for &p in &[0.001, 0.1, 0.5, 1.0, 1.5, 1.999] {
            assert_relative_eq!(erfc(erfc_inv(p)), p, epsilon = 1e-12);
        }
    }

    #[test]
    fn erf_inv_reference() {
        // erfinv(0.5) = 0.4769362762044699 (SciPy)
        assert_relative_eq!(erf_inv(0.5), 0.476_936_276_204_469_9, max_relative = 1e-12);
    }

    #[test]
    fn erf_inv_boundaries() {
        assert_eq!(erf_inv(1.0), f64::INFINITY);
        assert_eq!(erf_inv(-1.0), f64::NEG_INFINITY);
        assert!(erf_inv(1.5).is_nan());
        assert!(erf_inv(-1.5).is_nan());
    }
}
