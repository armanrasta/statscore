use rand::Rng;
use crate::error::Result;

/// Continuous distribution trait
pub trait ContinuousDistribution {
    fn pdf(&self, x: f64) -> f64;
    fn log_pdf(&self, x: f64) -> Option<f64> {
        let p = self.pdf(x);
        if p > 0.0 { Some(p.ln()) } else { None }
    }
    fn cdf(&self, x: f64) -> f64;
    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }
    fn ppf(&self, p: f64) -> Result<f64>;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<f64>;

    // Optional methods with default implementations
    fn hazard(&self, x: f64) -> Option<f64> {
        let sf = self.sf(x);
        if sf > 0.0 {
            Some(self.pdf(x) / sf)
        } else {
            None
        }
    }

    fn mean(&self) -> Option<f64> { None }
    fn variance(&self) -> Option<f64> { None }
    fn skewness(&self) -> Option<f64> { None }
    fn kurtosis(&self) -> Option<f64> { None }
    fn entropy(&self) -> Option<f64> { None }
}

/// Discrete distribution trait
pub trait DiscreteDistribution {
    fn pmf(&self, k: i64) -> f64;
    fn log_pmf(&self, k: i64) -> Option<f64> {
        let p = self.pmf(k);
        if p > 0.0 { Some(p.ln()) } else { None }
    }
    fn cdf(&self, k: i64) -> f64;
    fn sf(&self, k: i64) -> f64 {
        1.0 - self.cdf(k)
    }
    fn ppf(&self, p: f64) -> Result<i64>;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<i64>;

    fn mean(&self) -> Option<f64> { None }
    fn variance(&self) -> Option<f64> { None }
    fn skewness(&self) -> Option<f64> { None }
    fn kurtosis(&self) -> Option<f64> { None }
}

/// Marker trait for parameterized distributions
pub trait ParametricDistribution<P> {
    fn new(params: P) -> Result<Self> where Self: Sized;
    fn params(&self) -> &P;
}

/// Fittable distribution (MLE, method of moments)
pub trait FittableDistribution<D> {
    fn fit_mle(data: &[D]) -> Result<Self> where Self: Sized;
    fn fit_mom(data: &[D]) -> Result<Self> where Self: Sized;
}