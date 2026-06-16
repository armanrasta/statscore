pub mod distribution;
pub mod estimator;
pub mod test;

// Flat re-exports so callers write `use statscore_common::ContinuousDistribution`
pub use distribution::{
    ContinuousDistribution,
    DiscreteDistribution,
    FittableDistribution,
    MleFit,
    MomFit,
    MultivariateContinuousDistribution,
};
pub use estimator::{IntervalEstimator, ModelEstimator, PointEstimator};
pub use test::{Alternative, HypothesisTest, TestResult};