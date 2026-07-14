//! Helpers shared by distribution implementations.

use statscore_common::{Result, StatsError, require_in_range, require_positive};

/// Validate probability `p ∈ [0, 1]`.
pub(crate) fn require_prob(p: f64) -> Result<()> {
    require_in_range(p, 0.0, 1.0, "p")
}

/// Validate a strictly positive parameter.
pub(crate) fn require_pos(value: f64, name: &str) -> Result<()> {
    require_positive(value, name)
}

/// Validate `lo < hi` for a continuous interval support.
pub(crate) fn require_interval(lo: f64, hi: f64) -> Result<()> {
    if !(lo < hi) {
        return Err(StatsError::domain(format!(
            "interval requires lo < hi, got lo={lo}, hi={hi}"
        )));
    }
    if !lo.is_finite() || !hi.is_finite() {
        return Err(StatsError::domain("interval endpoints must be finite"));
    }
    Ok(())
}

/// Inverse of the regularized lower incomplete gamma P(a, ·) via Newton + bisection.
///
/// Returns `x ≥ 0` such that `gammainc(a, x) ≈ p`.
pub(crate) fn gammaincinv(a: f64, p: f64) -> f64 {
    use statscore_special::{gammainc, ln_gamma};

    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if a <= 0.0 {
        return f64::NAN;
    }

    // Wilks / Abramowitz initial guess
    let mut x = if a > 1.0 {
        let y = 2.257 * (1.0 - 2.0 / (9.0 * a) - (1.0 - p).ln().abs().sqrt()).powi(3) * a;
        y.max(0.1)
    } else {
        // Small shape: use power approximation
        (p * a.exp() * ln_gamma(a + 1.0).exp()).powf(1.0 / a)
    };
    if !x.is_finite() || x <= 0.0 {
        x = a;
    }

    let mut lo = 0.0_f64;
    let mut hi = (x * 10.0).max(a * 10.0).max(1.0);

    // Expand upper bound until CDF exceeds p
    while gammainc(a, hi) < p && hi < 1e300 {
        hi *= 2.0;
    }

    for _ in 0..200 {
        let err = gammainc(a, x) - p;
        if err.abs() < 1e-14 {
            break;
        }
        if err > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        // Newton using d/dx P(a,x) = x^{a-1} e^{-x} / Γ(a)
        let dens = (a - 1.0) * x.ln() - x - ln_gamma(a);
        let dens = dens.exp();
        let newton = if dens > 0.0 && dens.is_finite() {
            x - err / dens
        } else {
            f64::NAN
        };
        x = if newton.is_finite() && newton > lo && newton < hi {
            newton
        } else {
            0.5 * (lo + hi)
        };
    }
    x
}
