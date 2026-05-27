use crate::error::Result;

/// Point estimator
pub trait PointEstimator<T> {
    type Target;
    fn estimate(&self, data: &[T]) -> Result<Self::Target>;
    fn bias(&self, true_value: &Self::Target, estimated: &Self::Target) -> f64;
    fn mse(&self, true_value: &Self::Target, estimated: &Self::Target) -> f64;
}

/// Interval estimator
pub trait IntervalEstimator<T> {
    type Target;
    type Interval;

    fn confidence_interval(
        &self,
        data: &[T],
        confidence_level: f64,
    ) -> Result<Self::Interval>;
}

/// Regression-like estimator
pub trait ModelEstimator<I, O> {
    type Model;

    fn fit(&self, inputs: &[I], outputs: &[O]) -> Result<Self::Model>;
    fn predict(&self, model: &Self::Model, inputs: &[I]) -> Result<Vec<O>>;
}