//! Python wrappers for continuous and discrete distributions.
//!
//! Methods accept **Python floats** or **NumPy arrays** (and array-like sequences).
//! Scalars return `float`; arrays return `numpy.ndarray`.

use pyo3::prelude::*;
use rand::rng;
use statscore_common::{ContinuousDistribution, DiscreteDistribution};
use statscore_distributions::{
    Beta as RustBeta, Binomial as RustBinomial, ChiSquared as RustChiSquared,
    Exponential as RustExponential, FDistribution as RustF, Gamma as RustGamma,
    Geometric as RustGeometric, Normal as RustNormal, Poisson as RustPoisson,
    StudentT as RustStudentT, Uniform as RustUniform,
};

use crate::convert::{
    map_f64, map_f64_result, map_f64_to_i64_result, map_i64_to_f64, vec_f64_to_numpy,
    vec_i64_to_numpy,
};

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
                let inner = $ctor($($field),+).map_err(crate::error::stats_to_py)?;
                Ok(Self { inner })
            }

            /// Probability density. Accepts float or 1-D ndarray / array-like.
            fn pdf<'py>(
                &self,
                py: Python<'py>,
                x: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                map_f64(py, x, |v| ContinuousDistribution::pdf(&self.inner, v))
            }

            /// Log probability density. Accepts float or ndarray.
            fn logpdf<'py>(
                &self,
                py: Python<'py>,
                x: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                map_f64(py, x, |v| ContinuousDistribution::log_pdf(&self.inner, v))
            }

            /// Cumulative distribution. Accepts float or ndarray.
            fn cdf<'py>(
                &self,
                py: Python<'py>,
                x: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                map_f64(py, x, |v| ContinuousDistribution::cdf(&self.inner, v))
            }

            /// Survival `1 - cdf(x)`. Accepts float or ndarray.
            fn sf<'py>(
                &self,
                py: Python<'py>,
                x: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                map_f64(py, x, |v| ContinuousDistribution::sf(&self.inner, v))
            }

            /// Percent-point (quantile). Accepts float or ndarray.
            fn ppf<'py>(
                &self,
                py: Python<'py>,
                p: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                map_f64_result(py, p, |v| ContinuousDistribution::ppf(&self.inner, v))
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

            /// Draw `size` random samples as a 1-D `numpy.ndarray`.
            #[pyo3(signature = (size = 1))]
            fn rvs<'py>(&self, py: Python<'py>, size: usize) -> Bound<'py, numpy::PyArray1<f64>> {
                let samples = ContinuousDistribution::sample(&self.inner, &mut rng(), size);
                vec_f64_to_numpy(py, samples)
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
                let inner = $ctor($($field),+).map_err(crate::error::stats_to_py)?;
                Ok(Self { inner })
            }

            /// Probability mass. Accepts int or 1-D int/float ndarray.
            fn pmf<'py>(
                &self,
                py: Python<'py>,
                k: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                map_i64_to_f64(py, k, |v| DiscreteDistribution::pmf(&self.inner, v))
            }

            /// Log probability mass. Accepts int or ndarray.
            fn logpmf<'py>(
                &self,
                py: Python<'py>,
                k: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                map_i64_to_f64(py, k, |v| DiscreteDistribution::log_pmf(&self.inner, v))
            }

            /// Cumulative distribution. Accepts int or ndarray.
            fn cdf<'py>(
                &self,
                py: Python<'py>,
                k: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                map_i64_to_f64(py, k, |v| DiscreteDistribution::cdf(&self.inner, v))
            }

            /// Quantile (smallest k with cdf ≥ p). Accepts float or ndarray.
            fn ppf<'py>(
                &self,
                py: Python<'py>,
                p: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                map_f64_to_i64_result(py, p, |v| DiscreteDistribution::ppf(&self.inner, v))
            }

            fn mean(&self) -> f64 {
                DiscreteDistribution::mean(&self.inner)
            }

            fn var(&self) -> f64 {
                DiscreteDistribution::variance(&self.inner)
            }

            /// Draw `size` random samples as a 1-D `numpy.ndarray` of `int64`.
            #[pyo3(signature = (size = 1))]
            fn rvs<'py>(&self, py: Python<'py>, size: usize) -> Bound<'py, numpy::PyArray1<i64>> {
                let samples = DiscreteDistribution::sample(&self.inner, &mut rng(), size);
                vec_i64_to_numpy(py, samples)
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
