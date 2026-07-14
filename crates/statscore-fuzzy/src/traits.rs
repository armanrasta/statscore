//! Core fuzzy abstractions: [`FuzzySet`] and [`FuzzyNumber`].

use statscore_common::Result;

/// A fuzzy set defined by a membership function `μ: ℝ → [0, 1]`.
///
/// Every method describes the set in terms of that membership function:
/// - `membership(x)` is `μ(x)`.
/// - `core` is `{ x | μ(x) = 1 }`.
/// - `support` is the closure of `{ x | μ(x) > 0 }`.
/// - `alpha_cut(α)` is `{ x | μ(x) ≥ α }`.
pub trait FuzzySet {
    /// Membership degree `μ(x) ∈ [0, 1]` of the value `x`.
    fn membership(&self, x: f64) -> f64;

    /// The core: points whose membership equals `1`.
    fn core(&self) -> Vec<f64>;

    /// The support as an inclusive `(min, max)` interval.
    fn support(&self) -> (f64, f64);

    /// The `α`-cut `{ x | μ(x) ≥ α }` as an inclusive `(low, high)` interval.
    ///
    /// Returns `(NaN, NaN)` if `α` is outside `[0, 1]`.
    fn alpha_cut(&self, alpha: f64) -> (f64, f64);
}

/// A fuzzy number: a normal, convex fuzzy set on the real line that can be
/// collapsed to a single crisp value via *defuzzification*.
pub trait FuzzyNumber: FuzzySet {
    /// Defuzzify using the center of gravity (centroid).
    fn defuzzify_cog(&self) -> f64;

    /// Defuzzify using the mean of maxima (average of the core).
    fn defuzzify_mom(&self) -> f64;

    /// Defuzzify using a weighted blend of the mean-of-maxima and centroid.
    ///
    /// `weights[0]` scales the mean-of-maxima, `weights[1]` scales the
    /// centroid; the result is their normalized weighted average.
    ///
    /// # Errors
    /// Returns [`StatsError::InsufficientData`](statscore_common::StatsError::InsufficientData)
    /// if fewer than two weights are supplied, and
    /// [`StatsError::Domain`](statscore_common::StatsError::Domain) if the
    /// weights sum to zero.
    fn defuzzify_weighted(&self, weights: &[f64]) -> Result<f64>;
}
