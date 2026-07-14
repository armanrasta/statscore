//! Fuzzy logic operators: t-norms (AND), t-conorms (OR), complement, and a
//! Mamdani implication.
//!
//! All operators take membership degrees in `[0, 1]` and return a degree in
//! `[0, 1]`. Inputs are clamped to that range so downstream code never sees
//! out-of-range values.

/// Namespace for stateless fuzzy logic operators.
pub struct FuzzyLogic;

#[inline]
fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

impl FuzzyLogic {
    /// Fuzzy AND — minimum t-norm (the most common choice).
    #[must_use]
    pub fn fuzzy_and_min(mu_a: f64, mu_b: f64) -> f64 {
        clamp01(mu_a).min(clamp01(mu_b))
    }

    /// Fuzzy AND — algebraic product t-norm.
    #[must_use]
    pub fn fuzzy_and_product(mu_a: f64, mu_b: f64) -> f64 {
        clamp01(mu_a) * clamp01(mu_b)
    }

    /// Fuzzy AND — Łukasiewicz t-norm `max(0, a + b − 1)`.
    #[must_use]
    pub fn fuzzy_and_lukasiewicz(mu_a: f64, mu_b: f64) -> f64 {
        (clamp01(mu_a) + clamp01(mu_b) - 1.0).max(0.0)
    }

    /// Fuzzy OR — maximum t-conorm.
    #[must_use]
    pub fn fuzzy_or_max(mu_a: f64, mu_b: f64) -> f64 {
        clamp01(mu_a).max(clamp01(mu_b))
    }

    /// Fuzzy OR — algebraic (probabilistic) sum `a + b − a·b`.
    #[must_use]
    pub fn fuzzy_or_sum(mu_a: f64, mu_b: f64) -> f64 {
        let (a, b) = (clamp01(mu_a), clamp01(mu_b));
        a + b - a * b
    }

    /// Fuzzy NOT — standard complement `1 − μ`.
    #[must_use]
    pub fn fuzzy_not(mu: f64) -> f64 {
        1.0 - clamp01(mu)
    }

    /// Mamdani implication `IF A THEN B`, modeled as the product of the
    /// antecedent and consequent degrees.
    #[must_use]
    pub fn implication(mu_condition: f64, mu_consequence: f64) -> f64 {
        clamp01(mu_condition) * clamp01(mu_consequence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn and_operators() {
        assert!((FuzzyLogic::fuzzy_and_min(0.7, 0.8) - 0.7).abs() < EPS);
        assert!((FuzzyLogic::fuzzy_and_product(0.7, 0.8) - 0.56).abs() < EPS);
        assert!((FuzzyLogic::fuzzy_and_lukasiewicz(0.7, 0.8) - 0.5).abs() < EPS);
        assert_eq!(FuzzyLogic::fuzzy_and_lukasiewicz(0.2, 0.3), 0.0);
    }

    #[test]
    fn or_operators() {
        assert!((FuzzyLogic::fuzzy_or_max(0.7, 0.8) - 0.8).abs() < EPS);
        assert!((FuzzyLogic::fuzzy_or_sum(0.7, 0.8) - 0.94).abs() < EPS);
    }

    #[test]
    fn not_is_involutive() {
        for mu in [0.0, 0.3, 0.5, 0.7, 1.0] {
            let round = FuzzyLogic::fuzzy_not(FuzzyLogic::fuzzy_not(mu));
            assert!((round - mu).abs() < EPS);
        }
    }

    #[test]
    fn de_morgan_min_max() {
        // NOT(A AND B) == (NOT A) OR (NOT B) for min/max pair.
        let (a, b) = (0.3, 0.75);
        let lhs = FuzzyLogic::fuzzy_not(FuzzyLogic::fuzzy_and_min(a, b));
        let rhs = FuzzyLogic::fuzzy_or_max(FuzzyLogic::fuzzy_not(a), FuzzyLogic::fuzzy_not(b));
        assert!((lhs - rhs).abs() < EPS);
    }

    #[test]
    fn inputs_are_clamped() {
        assert_eq!(FuzzyLogic::fuzzy_and_min(1.5, -0.2), 0.0);
        assert_eq!(FuzzyLogic::fuzzy_or_max(1.5, -0.2), 1.0);
    }
}
