//! Python wrappers for continuous and discrete distributions.

use pyo3::prelude::*;
use rand::rng;
use statscore_common::{ContinuousDistribution, DiscreteDistribution};
use statscore_distributions::{
    Beta as RustBeta, Binomial as RustBinomial, ChiSquared as RustChiSquared,
    Exponential as RustExponential, FDistribution as RustF, Gamma as RustGamma,
    Geometric as RustGeometric, Normal as RustNormal, Poisson as RustPoisson,
    StudentT as RustStudentT, Uniform as RustUniform,
};

use crate::error::stats_to_py;

macro_rules! wrap_continuous {
    ($pyname:ident, $rust:ty, $ctor:path, $($field:ident : $fty:ty),+) => {
        #[pyclass(module = "statscore.distributions", skip_from_py_object)]
        #[derive(Clone)]
        pub struct $pyname {
            inner: $rust,
        }

        #[pymethods]
        impl $pyname {
            #[new]
            fn new($($field: $fty),+) -> PyResult<Self> {
                let inner = $ctor($($field),+).map_err(stats_to_py)?;
                Ok(Self { inner })
            }

            /// Probability density function.
            fn pdf(&self, x: f64) -> f64 {
                ContinuousDistribution::pdf(&self.inner, x)
            }

            /// Log probability density.
            fn logpdf(&self, x: f64) -> f64 {
                ContinuousDistribution::log_pdf(&self.inner, x)
            }

            /// Cumulative distribution function.
            fn cdf(&self, x: f64) -> f64 {
                ContinuousDistribution::cdf(&self.inner, x)
            }

            /// Survival function `1 - cdf(x)`.
            fn sf(&self, x: f64) -> f64 {
                ContinuousDistribution::sf(&self.inner, x)
            }

            /// Percent-point (quantile) function.
            fn ppf(&self, p: f64) -> PyResult<f64> {
                ContinuousDistribution::ppf(&self.inner, p).map_err(stats_to_py)
            }

            /// Mean (may be NaN if undefined).
            fn mean(&self) -> f64 {
                ContinuousDistribution::mean(&self.inner)
            }

            /// Variance (may be NaN if undefined).
            fn var(&self) -> f64 {
                ContinuousDistribution::variance(&self.inner)
            }

            /// Standard deviation.
            fn std(&self) -> f64 {
                ContinuousDistribution::std_dev(&self.inner)
            }

            /// Draw `size` random samples.
            #[pyo3(signature = (size = 1))]
            fn rvs(&self, size: usize) -> Vec<f64> {
                ContinuousDistribution::sample(&self.inner, &mut rng(), size)
            }

            fn __repr__(&self) -> String {
                format!("{:?}", self.inner)
            }
        }
    };
}

wrap_continuous!(Normal, RustNormal, RustNormal::new, loc: f64, scale: f64);
wrap_continuous!(Uniform, RustUniform, RustUniform::new, a: f64, b: f64);
wrap_continuous!(Exponential, RustExponential, RustExponential::new, rate: f64);
wrap_continuous!(Gamma, RustGamma, RustGamma::new, shape: f64, scale: f64);
wrap_continuous!(Beta, RustBeta, RustBeta::new, alpha: f64, beta: f64);
wrap_continuous!(ChiSquared, RustChiSquared, RustChiSquared::new, df: f64);
wrap_continuous!(StudentT, RustStudentT, RustStudentT::new, df: f64);
wrap_continuous!(F, RustF, RustF::new, dfn: f64, dfd: f64);

macro_rules! wrap_discrete {
    ($pyname:ident, $rust:ty, $ctor:path, $($field:ident : $fty:ty),+) => {
        #[pyclass(module = "statscore.distributions", skip_from_py_object)]
        #[derive(Clone)]
        pub struct $pyname {
            inner: $rust,
        }

        #[pymethods]
        impl $pyname {
            #[new]
            fn new($($field: $fty),+) -> PyResult<Self> {
                let inner = $ctor($($field),+).map_err(stats_to_py)?;
                Ok(Self { inner })
            }

            /// Probability mass function.
            fn pmf(&self, k: i64) -> f64 {
                DiscreteDistribution::pmf(&self.inner, k)
            }

            /// Log probability mass.
            fn logpmf(&self, k: i64) -> f64 {
                DiscreteDistribution::log_pmf(&self.inner, k)
            }

            /// Cumulative distribution function.
            fn cdf(&self, k: i64) -> f64 {
                DiscreteDistribution::cdf(&self.inner, k)
            }

            /// Quantile function (smallest k with cdf ≥ p).
            fn ppf(&self, p: f64) -> PyResult<i64> {
                DiscreteDistribution::ppf(&self.inner, p).map_err(stats_to_py)
            }

            fn mean(&self) -> f64 {
                DiscreteDistribution::mean(&self.inner)
            }

            fn var(&self) -> f64 {
                DiscreteDistribution::variance(&self.inner)
            }

            #[pyo3(signature = (size = 1))]
            fn rvs(&self, size: usize) -> Vec<i64> {
                DiscreteDistribution::sample(&self.inner, &mut rng(), size)
            }

            fn __repr__(&self) -> String {
                format!("{:?}", self.inner)
            }
        }
    };
}

wrap_discrete!(Binomial, RustBinomial, RustBinomial::new, n: u64, p: f64);
wrap_discrete!(Poisson, RustPoisson, RustPoisson::new, lambda: f64);
wrap_discrete!(Geometric, RustGeometric, RustGeometric::new, p: f64);

/// Register distribution types on the `distributions` submodule.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Normal>()?;
    module.add_class::<Uniform>()?;
    module.add_class::<Exponential>()?;
    module.add_class::<Gamma>()?;
    module.add_class::<Beta>()?;
    module.add_class::<ChiSquared>()?;
    module.add_class::<StudentT>()?;
    module.add_class::<F>()?;
    module.add_class::<Binomial>()?;
    module.add_class::<Poisson>()?;
    module.add_class::<Geometric>()?;

    #[pyfunction]
    fn standard_normal() -> Normal {
        Normal {
            inner: RustNormal::standard(),
        }
    }
    module.add_function(wrap_pyfunction!(standard_normal, module)?)?;
    Ok(())
}
