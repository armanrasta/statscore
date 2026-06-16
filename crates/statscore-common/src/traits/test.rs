//! Hypothesis test traits and result types.
//!
//! ## Naming consistency
//! The field is `pvalue` (no underscore) to match SciPy, R broom, and
//! the Python binding layer (`result["pvalue"]`).

use crate::error::Result;

/// Direction of the alternative hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Alternative {
    /// H₁: parameter ≠ null value (default).
    TwoSided,
    /// H₁: parameter < null value.
    Less,
    /// H₁: parameter > null value.
    Greater,
}

impl Default for Alternative {
    fn default() -> Self {
        Self::TwoSided
    }
}

impl Alternative {
    /// Parse from a string, accepting both formats ("two-sided" and "two_sided").
    ///
    /// # Errors
    /// Returns an error string if the value is unrecognised.
    pub fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.to_lowercase().replace('-', "_").as_str() {
            "two_sided" | "twosided" | "two"     => Ok(Self::TwoSided),
            "less"      | "left"     | "lower"   => Ok(Self::Less),
            "greater"   | "right"    | "upper"   => Ok(Self::Greater),
            other => Err(format!(
                "Unknown alternative '{other}'. \
                 Use 'two-sided', 'less', or 'greater'."
            )),
        }
    }
}

/// The result returned by any hypothesis test.
///
/// ## Fields
/// - `statistic` — the test statistic value
/// - `pvalue`    — the p-value (note: no underscore)
/// - `df`        — degrees of freedom (if applicable)
/// - `alternative` — which alternative hypothesis was tested
/// - `conf_int`  — confidence interval (if computed)
/// - `effect_size` — effect size estimate (if computed)
/// - `null_value` — the null hypothesis value (e.g., μ₀ in a t-test)
#[derive(Debug, Clone)]
pub struct TestResult {
    /// The computed test statistic (e.g., t, F, χ², z).
    pub statistic: f64,

    /// The p-value associated with the test statistic.
    ///
    /// Always in `[0, 1]`. Computed under the assumption that H₀ is true.
    pub pvalue: f64,

    /// Degrees of freedom (if the test statistic has a named distribution).
    ///
    /// `None` for tests where df is not applicable (e.g., exact tests).
    pub df: Option<f64>,

    /// The alternative hypothesis direction that was tested.
    pub alternative: Alternative,

    /// Confidence interval for the parameter of interest.
    ///
    /// `None` if not computed or not applicable.
    pub conf_int: Option<(f64, f64)>,

    /// Standardised effect size (e.g., Cohen's d for t-tests).
    ///
    /// `None` if not computed.
    pub effect_size: Option<f64>,

    /// The null hypothesis value (e.g., `0.0` in a one-sample t-test for μ = 0).
    pub null_value: Option<f64>,
}

impl TestResult {
    /// Convenience constructor for the common case (statistic + pvalue only).
    pub fn new(statistic: f64, pvalue: f64) -> Self {
        Self {
            statistic,
            pvalue,
            df: None,
            alternative: Alternative::TwoSided,
            conf_int: None,
            effect_size: None,
            null_value: None,
        }
    }

    /// Builder: attach degrees of freedom.
    pub fn with_df(mut self, df: f64) -> Self {
        self.df = Some(df);
        self
    }

    /// Builder: attach the alternative hypothesis.
    pub fn with_alternative(mut self, alt: Alternative) -> Self {
        self.alternative = alt;
        self
    }

    /// Builder: attach a confidence interval.
    pub fn with_conf_int(mut self, lo: f64, hi: f64) -> Self {
        self.conf_int = Some((lo, hi));
        self
    }

    /// Builder: attach an effect size.
    pub fn with_effect_size(mut self, es: f64) -> Self {
        self.effect_size = Some(es);
        self
    }

    /// Builder: attach the null value.
    pub fn with_null_value(mut self, v: f64) -> Self {
        self.null_value = Some(v);
        self
    }

    /// Returns `true` if `pvalue < alpha` (reject H₀).
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.pvalue < alpha
    }
}

/// A hypothesis test that operates on a given input type.
///
/// # Type parameters
/// - `Input`  — the data structure fed to the test (e.g., `(&[f64], f64)`)
/// - `Config` — test configuration (e.g., `Alternative`, significance level)
pub trait HypothesisTest {
    type Input;
    type Config;

    /// Run the test and return a [`TestResult`].
    ///
    /// # Errors
    /// - [`crate::error::StatsError::InsufficientData`] if data is too small.
    /// - [`crate::error::StatsError::Domain`] if inputs are invalid.
    fn run(&self, data: &Self::Input, config: &Self::Config) -> Result<TestResult>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alternative_parse_variants() {
        assert_eq!(Alternative::from_str("two-sided").unwrap(), Alternative::TwoSided);
        assert_eq!(Alternative::from_str("two_sided").unwrap(), Alternative::TwoSided);
        assert_eq!(Alternative::from_str("less").unwrap(),      Alternative::Less);
        assert_eq!(Alternative::from_str("greater").unwrap(),   Alternative::Greater);
        assert_eq!(Alternative::from_str("left").unwrap(),      Alternative::Less);
        assert_eq!(Alternative::from_str("right").unwrap(),     Alternative::Greater);
        assert!(Alternative::from_str("bad").is_err());
    }

    #[test]
    fn test_result_builder() {
        let r = TestResult::new(2.5, 0.012)
            .with_df(18.0)
            .with_alternative(Alternative::TwoSided)
            .with_conf_int(0.1, 1.9)
            .with_effect_size(0.58)
            .with_null_value(0.0);

        assert!((r.statistic - 2.5).abs() < 1e-12);
        assert!((r.pvalue   - 0.012).abs() < 1e-12);
        assert_eq!(r.df,           Some(18.0));
        assert_eq!(r.conf_int,     Some((0.1, 1.9)));
        assert_eq!(r.effect_size,  Some(0.58));
        assert_eq!(r.null_value,   Some(0.0));
        assert!(r.is_significant(0.05));
        assert!(!r.is_significant(0.01));
    }
}