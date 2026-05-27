use crate::error::Result;

#[derive(Debug, Clone, PartialEq)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

impl Default for Alternative {
    fn default() -> Self {
        Alternative::TwoSided
    }
}

#[derive(Debug, Clone)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub alternative: Alternative,
    pub confidence_interval: Option<(f64, f64)>,
    pub effect_size: Option<f64>,
    pub null_value: Option<f64>,
}

/// Hypothesis test trait
pub trait HypothesisTest {
    type Input;
    type Config;

    fn run(&self, data: &Self::Input, config: &Self::Config) -> Result<TestResult>;
}