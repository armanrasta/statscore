//! Concrete fuzzy numbers: triangular and trapezoidal.

use statscore_common::{Result, StatsError};

use crate::traits::{FuzzyNumber, FuzzySet};

/// Triangular fuzzy number with vertices `a < m < b` and peak `μ(m) = 1`.
///
/// The membership rises linearly from `a` to `m` and falls linearly from
/// `m` to `b`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangularFuzzyNumber {
    /// Left boundary (`μ(a) = 0`).
    pub a: f64,
    /// Peak (`μ(m) = 1`).
    pub m: f64,
    /// Right boundary (`μ(b) = 0`).
    pub b: f64,
}

impl TriangularFuzzyNumber {
    /// Create a triangular fuzzy number.
    ///
    /// # Errors
    /// Returns [`StatsError::Domain`] unless `a < m < b` and all vertices are
    /// finite.
    ///
    /// # Example
    /// ```
    /// use statscore_fuzzy::sets::TriangularFuzzyNumber;
    /// let t = TriangularFuzzyNumber::new(1.0, 2.0, 3.0).unwrap();
    /// assert!(TriangularFuzzyNumber::new(3.0, 2.0, 1.0).is_err());
    /// # let _ = t;
    /// ```
    pub fn new(a: f64, m: f64, b: f64) -> Result<Self> {
        if ![a, m, b].iter().all(|v| v.is_finite()) {
            return Err(StatsError::domain(
                "triangular fuzzy number: vertices must be finite",
            ));
        }
        if a >= m || m >= b {
            return Err(StatsError::domain(
                "triangular fuzzy number: require a < m < b",
            ));
        }
        Ok(Self { a, m, b })
    }
}

impl FuzzySet for TriangularFuzzyNumber {
    fn membership(&self, x: f64) -> f64 {
        if x.is_nan() || x < self.a || x > self.b {
            0.0
        } else if x <= self.m {
            (x - self.a) / (self.m - self.a)
        } else {
            (self.b - x) / (self.b - self.m)
        }
    }

    fn core(&self) -> Vec<f64> {
        vec![self.m]
    }

    fn support(&self) -> (f64, f64) {
        (self.a, self.b)
    }

    fn alpha_cut(&self, alpha: f64) -> (f64, f64) {
        if !(0.0..=1.0).contains(&alpha) {
            return (f64::NAN, f64::NAN);
        }
        let left = self.a + alpha * (self.m - self.a);
        let right = self.b - alpha * (self.b - self.m);
        (left, right)
    }
}

impl FuzzyNumber for TriangularFuzzyNumber {
    fn defuzzify_cog(&self) -> f64 {
        (self.a + self.m + self.b) / 3.0
    }

    fn defuzzify_mom(&self) -> f64 {
        self.m
    }

    fn defuzzify_weighted(&self, weights: &[f64]) -> Result<f64> {
        weighted_blend(self.defuzzify_mom(), self.defuzzify_cog(), weights)
    }
}

/// Trapezoidal fuzzy number with vertices `a < m1 ≤ m2 < b` and a flat top
/// `μ(x) = 1` for `x ∈ [m1, m2]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrapezoidalFuzzyNumber {
    /// Left boundary (`μ(a) = 0`).
    pub a: f64,
    /// Left shoulder (`μ(m1) = 1`).
    pub m1: f64,
    /// Right shoulder (`μ(m2) = 1`).
    pub m2: f64,
    /// Right boundary (`μ(b) = 0`).
    pub b: f64,
}

impl TrapezoidalFuzzyNumber {
    /// Create a trapezoidal fuzzy number.
    ///
    /// # Errors
    /// Returns [`StatsError::Domain`] unless `a < m1 ≤ m2 < b` and all vertices
    /// are finite.
    ///
    /// # Example
    /// ```
    /// use statscore_fuzzy::sets::TrapezoidalFuzzyNumber;
    /// let t = TrapezoidalFuzzyNumber::new(0.0, 1.0, 3.0, 4.0).unwrap();
    /// # let _ = t;
    /// ```
    pub fn new(a: f64, m1: f64, m2: f64, b: f64) -> Result<Self> {
        if ![a, m1, m2, b].iter().all(|v| v.is_finite()) {
            return Err(StatsError::domain(
                "trapezoidal fuzzy number: vertices must be finite",
            ));
        }
        if a >= m1 || m1 > m2 || m2 >= b {
            return Err(StatsError::domain(
                "trapezoidal fuzzy number: require a < m1 <= m2 < b",
            ));
        }
        Ok(Self { a, m1, m2, b })
    }
}

impl FuzzySet for TrapezoidalFuzzyNumber {
    fn membership(&self, x: f64) -> f64 {
        if x.is_nan() || x < self.a || x > self.b {
            0.0
        } else if x < self.m1 {
            (x - self.a) / (self.m1 - self.a)
        } else if x <= self.m2 {
            1.0
        } else {
            (self.b - x) / (self.b - self.m2)
        }
    }

    fn core(&self) -> Vec<f64> {
        vec![self.m1, self.m2]
    }

    fn support(&self) -> (f64, f64) {
        (self.a, self.b)
    }

    fn alpha_cut(&self, alpha: f64) -> (f64, f64) {
        if !(0.0..=1.0).contains(&alpha) {
            return (f64::NAN, f64::NAN);
        }
        let left = self.a + alpha * (self.m1 - self.a);
        let right = self.b - alpha * (self.b - self.m2);
        (left, right)
    }
}

impl FuzzyNumber for TrapezoidalFuzzyNumber {
    fn defuzzify_cog(&self) -> f64 {
        // Centroid of a trapezoid (closed form). Reduces to the triangular
        // centroid when m1 == m2.
        let (a, b, c, d) = (self.a, self.m1, self.m2, self.b);
        let num = (d + c).powi(2) - d * c - (a + b).powi(2) + a * b;
        let den = 3.0 * ((d + c) - (a + b));
        if den.abs() < f64::EPSILON {
            (a + b + c + d) / 4.0
        } else {
            num / den
        }
    }

    fn defuzzify_mom(&self) -> f64 {
        (self.m1 + self.m2) / 2.0
    }

    fn defuzzify_weighted(&self, weights: &[f64]) -> Result<f64> {
        weighted_blend(self.defuzzify_mom(), self.defuzzify_cog(), weights)
    }
}

/// Normalized weighted average of the mean-of-maxima and centroid estimates.
fn weighted_blend(mom: f64, cog: f64, weights: &[f64]) -> Result<f64> {
    statscore_common::require_min_len(weights, 2)?;
    let (w1, w2) = (weights[0], weights[1]);
    let total = w1 + w2;
    if total.abs() < f64::EPSILON {
        return Err(StatsError::domain(
            "defuzzify_weighted: weights must not sum to zero",
        ));
    }
    Ok((w1 * mom + w2 * cog) / total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangular_membership() {
        let t = TriangularFuzzyNumber::new(1.0, 2.0, 3.0).unwrap();
        assert_eq!(t.membership(2.0), 1.0);
        assert_eq!(t.membership(1.5), 0.5);
        assert_eq!(t.membership(2.5), 0.5);
        assert_eq!(t.membership(0.0), 0.0);
        assert_eq!(t.membership(4.0), 0.0);
    }

    #[test]
    fn triangular_invalid() {
        assert!(TriangularFuzzyNumber::new(2.0, 2.0, 3.0).is_err());
        assert!(TriangularFuzzyNumber::new(1.0, 3.0, 2.0).is_err());
        assert!(TriangularFuzzyNumber::new(f64::NAN, 2.0, 3.0).is_err());
    }

    #[test]
    fn triangular_alpha_cut() {
        let t = TriangularFuzzyNumber::new(0.0, 1.0, 2.0).unwrap();
        assert_eq!(t.alpha_cut(0.0), (0.0, 2.0));
        assert_eq!(t.alpha_cut(1.0), (1.0, 1.0));
        assert_eq!(t.alpha_cut(0.5), (0.5, 1.5));
        assert!(t.alpha_cut(1.5).0.is_nan());
    }

    #[test]
    fn triangular_defuzzify() {
        let t = TriangularFuzzyNumber::new(1.0, 2.0, 6.0).unwrap();
        assert!((t.defuzzify_cog() - 3.0).abs() < 1e-12);
        assert_eq!(t.defuzzify_mom(), 2.0);
        assert!((t.defuzzify_weighted(&[1.0, 1.0]).unwrap() - 2.5).abs() < 1e-12);
        assert!(t.defuzzify_weighted(&[0.0, 0.0]).is_err());
        assert!(t.defuzzify_weighted(&[1.0]).is_err());
    }

    #[test]
    fn trapezoidal_membership() {
        let t = TrapezoidalFuzzyNumber::new(0.0, 1.0, 3.0, 4.0).unwrap();
        assert_eq!(t.membership(2.0), 1.0);
        assert_eq!(t.membership(1.0), 1.0);
        assert_eq!(t.membership(3.0), 1.0);
        assert_eq!(t.membership(0.5), 0.5);
        assert_eq!(t.membership(3.5), 0.5);
        assert_eq!(t.membership(-1.0), 0.0);
    }

    #[test]
    fn trapezoidal_symmetric_centroid() {
        let t = TrapezoidalFuzzyNumber::new(0.0, 1.0, 3.0, 4.0).unwrap();
        assert!((t.defuzzify_cog() - 2.0).abs() < 1e-12);
        assert_eq!(t.defuzzify_mom(), 2.0);
    }

    #[test]
    fn trapezoidal_reduces_to_triangle() {
        // m1 == m2 makes the trapezoid a triangle; centroids must agree.
        let trap = TrapezoidalFuzzyNumber::new(1.0, 2.0, 2.0, 6.0).unwrap();
        let tri = TriangularFuzzyNumber::new(1.0, 2.0, 6.0).unwrap();
        assert!((trap.defuzzify_cog() - tri.defuzzify_cog()).abs() < 1e-10);
    }

    #[test]
    fn support_contains_core() {
        let t = TrapezoidalFuzzyNumber::new(0.0, 1.0, 3.0, 4.0).unwrap();
        let (lo, hi) = t.support();
        for c in t.core() {
            assert!(lo <= c && c <= hi);
        }
    }
}
