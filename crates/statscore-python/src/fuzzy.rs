//! Python wrappers for fuzzy sets, logic, and statistics.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use statscore_fuzzy::{
    FuzzyLogic as RustLogic, FuzzyNumber, FuzzySet, TrapezoidalFuzzyNumber as RustTrap,
    TriangularFuzzyNumber as RustTri, fuzzy_correlation as rust_corr,
    fuzzy_mean as rust_mean, fuzzy_variance as rust_var,
};

use crate::convert::map_f64;
use crate::error::stats_to_py;

/// Triangular fuzzy number with vertices `a < m < b` and peak `μ(m) = 1`.
#[pyclass(module = "statscore.fuzzy", skip_from_py_object, name = "TriangularFuzzyNumber")]
#[derive(Clone)]
pub struct TriangularFuzzyNumber {
    inner: RustTri,
}

#[pymethods]
impl TriangularFuzzyNumber {
    #[new]
    fn new(a: f64, m: f64, b: f64) -> PyResult<Self> {
        let inner = RustTri::new(a, m, b).map_err(stats_to_py)?;
        Ok(Self { inner })
    }

    /// Left boundary (`μ(a) = 0`).
    #[getter]
    fn a(&self) -> f64 {
        self.inner.a
    }

    /// Peak (`μ(m) = 1`).
    #[getter]
    fn m(&self) -> f64 {
        self.inner.m
    }

    /// Right boundary (`μ(b) = 0`).
    #[getter]
    fn b(&self) -> f64 {
        self.inner.b
    }

    /// Membership `μ(x) ∈ [0, 1]`. Accepts float or 1-D ndarray.
    fn membership<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        map_f64(py, x, |v| FuzzySet::membership(&self.inner, v))
    }

    /// Core: points with `μ(x) = 1`.
    fn core(&self) -> Vec<f64> {
        FuzzySet::core(&self.inner)
    }

    /// Support as `(min, max)`.
    fn support<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let (lo, hi) = FuzzySet::support(&self.inner);
        Ok(PyTuple::new(py, [lo, hi])?)
    }

    /// Alpha-cut `{ x | μ(x) ≥ α }` as `(low, high)`.
    fn alpha_cut<'py>(&self, py: Python<'py>, alpha: f64) -> PyResult<Bound<'py, PyTuple>> {
        let (lo, hi) = FuzzySet::alpha_cut(&self.inner, alpha);
        Ok(PyTuple::new(py, [lo, hi])?)
    }

    /// Defuzzify via center of gravity (centroid).
    fn defuzzify_cog(&self) -> f64 {
        FuzzyNumber::defuzzify_cog(&self.inner)
    }

    /// Defuzzify via mean of maxima (peak).
    fn defuzzify_mom(&self) -> f64 {
        FuzzyNumber::defuzzify_mom(&self.inner)
    }

    /// Weighted blend of MOM and COG. Needs at least two weights.
    fn defuzzify_weighted(&self, weights: Vec<f64>) -> PyResult<f64> {
        FuzzyNumber::defuzzify_weighted(&self.inner, &weights).map_err(stats_to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "TriangularFuzzyNumber(a={}, m={}, b={})",
            self.inner.a, self.inner.m, self.inner.b
        )
    }
}

/// Trapezoidal fuzzy number with vertices `a < m1 ≤ m2 < b`.
#[pyclass(module = "statscore.fuzzy", skip_from_py_object, name = "TrapezoidalFuzzyNumber")]
#[derive(Clone)]
pub struct TrapezoidalFuzzyNumber {
    inner: RustTrap,
}

#[pymethods]
impl TrapezoidalFuzzyNumber {
    #[new]
    fn new(a: f64, m1: f64, m2: f64, b: f64) -> PyResult<Self> {
        let inner = RustTrap::new(a, m1, m2, b).map_err(stats_to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn a(&self) -> f64 {
        self.inner.a
    }

    #[getter]
    fn m1(&self) -> f64 {
        self.inner.m1
    }

    #[getter]
    fn m2(&self) -> f64 {
        self.inner.m2
    }

    #[getter]
    fn b(&self) -> f64 {
        self.inner.b
    }

    /// Membership `μ(x)`. Accepts float or 1-D ndarray.
    fn membership<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        map_f64(py, x, |v| FuzzySet::membership(&self.inner, v))
    }

    fn core(&self) -> Vec<f64> {
        FuzzySet::core(&self.inner)
    }

    fn support<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let (lo, hi) = FuzzySet::support(&self.inner);
        Ok(PyTuple::new(py, [lo, hi])?)
    }

    fn alpha_cut<'py>(&self, py: Python<'py>, alpha: f64) -> PyResult<Bound<'py, PyTuple>> {
        let (lo, hi) = FuzzySet::alpha_cut(&self.inner, alpha);
        Ok(PyTuple::new(py, [lo, hi])?)
    }

    fn defuzzify_cog(&self) -> f64 {
        FuzzyNumber::defuzzify_cog(&self.inner)
    }

    fn defuzzify_mom(&self) -> f64 {
        FuzzyNumber::defuzzify_mom(&self.inner)
    }

    fn defuzzify_weighted(&self, weights: Vec<f64>) -> PyResult<f64> {
        FuzzyNumber::defuzzify_weighted(&self.inner, &weights).map_err(stats_to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "TrapezoidalFuzzyNumber(a={}, m1={}, m2={}, b={})",
            self.inner.a, self.inner.m1, self.inner.m2, self.inner.b
        )
    }
}

/// Fuzzy AND — minimum t-norm.
#[pyfunction]
fn fuzzy_and_min(mu_a: f64, mu_b: f64) -> f64 {
    RustLogic::fuzzy_and_min(mu_a, mu_b)
}

/// Fuzzy AND — algebraic product.
#[pyfunction]
fn fuzzy_and_product(mu_a: f64, mu_b: f64) -> f64 {
    RustLogic::fuzzy_and_product(mu_a, mu_b)
}

/// Fuzzy AND — Łukasiewicz t-norm.
#[pyfunction]
fn fuzzy_and_lukasiewicz(mu_a: f64, mu_b: f64) -> f64 {
    RustLogic::fuzzy_and_lukasiewicz(mu_a, mu_b)
}

/// Fuzzy OR — maximum t-conorm.
#[pyfunction]
fn fuzzy_or_max(mu_a: f64, mu_b: f64) -> f64 {
    RustLogic::fuzzy_or_max(mu_a, mu_b)
}

/// Fuzzy OR — algebraic sum.
#[pyfunction]
fn fuzzy_or_sum(mu_a: f64, mu_b: f64) -> f64 {
    RustLogic::fuzzy_or_sum(mu_a, mu_b)
}

/// Fuzzy NOT — complement `1 − μ`.
#[pyfunction]
fn fuzzy_not(mu: f64) -> f64 {
    RustLogic::fuzzy_not(mu)
}

/// Mamdani implication.
#[pyfunction]
fn implication(mu_condition: f64, mu_consequence: f64) -> f64 {
    RustLogic::implication(mu_condition, mu_consequence)
}

/// Vertex-wise fuzzy mean of triangular fuzzy numbers.
#[pyfunction]
fn fuzzy_mean(values: Vec<PyRef<'_, TriangularFuzzyNumber>>) -> PyResult<TriangularFuzzyNumber> {
    let rust: Vec<_> = values.iter().map(|v| v.inner).collect();
    let mean = rust_mean(&rust).map_err(stats_to_py)?;
    Ok(TriangularFuzzyNumber { inner: mean })
}

/// Population variance of defuzzified (COG) values.
#[pyfunction]
fn fuzzy_variance(values: Vec<PyRef<'_, TriangularFuzzyNumber>>) -> PyResult<f64> {
    let rust: Vec<_> = values.iter().map(|v| v.inner).collect();
    rust_var(&rust).map_err(stats_to_py)
}

/// Pearson correlation of defuzzified values.
#[pyfunction]
fn fuzzy_correlation(
    x: Vec<PyRef<'_, TriangularFuzzyNumber>>,
    y: Vec<PyRef<'_, TriangularFuzzyNumber>>,
) -> PyResult<f64> {
    let xr: Vec<_> = x.iter().map(|v| v.inner).collect();
    let yr: Vec<_> = y.iter().map(|v| v.inner).collect();
    rust_corr(&xr, &yr).map_err(stats_to_py)
}

/// Register fuzzy types and functions on the `fuzzy` submodule.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<TriangularFuzzyNumber>()?;
    module.add_class::<TrapezoidalFuzzyNumber>()?;
    module.add_function(wrap_pyfunction!(fuzzy_and_min, module)?)?;
    module.add_function(wrap_pyfunction!(fuzzy_and_product, module)?)?;
    module.add_function(wrap_pyfunction!(fuzzy_and_lukasiewicz, module)?)?;
    module.add_function(wrap_pyfunction!(fuzzy_or_max, module)?)?;
    module.add_function(wrap_pyfunction!(fuzzy_or_sum, module)?)?;
    module.add_function(wrap_pyfunction!(fuzzy_not, module)?)?;
    module.add_function(wrap_pyfunction!(implication, module)?)?;
    module.add_function(wrap_pyfunction!(fuzzy_mean, module)?)?;
    module.add_function(wrap_pyfunction!(fuzzy_variance, module)?)?;
    module.add_function(wrap_pyfunction!(fuzzy_correlation, module)?)?;
    Ok(())
}
