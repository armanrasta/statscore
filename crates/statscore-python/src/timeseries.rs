//! Python wrappers for time series forecasting and diagnostics.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use rand::rng;
use statscore_timeseries::acf::{acf as rust_acf, pacf as rust_pacf};
use statscore_timeseries::arima::ArimaModel as RustArima;
use statscore_timeseries::baselines::{
    drift as rust_drift, naive as rust_naive, seasonal_naive as rust_seasonal_naive,
};
use statscore_timeseries::decomposition::{
    classical_decompose as rust_decompose, DecomposeModel as RustDecompose,
};
use statscore_timeseries::ets::{EtsFit as RustEts, EtsModel as RustEtsModel};
use statscore_timeseries::forecast::{Forecast as RustForecast, Forecaster};
use statscore_timeseries::markov::MarkovChain as RustMarkov;
use statscore_timeseries::prophet::{
    ProphetStyleModel as RustProphet, ProphetStyleSpec as RustProphetSpec,
};
use statscore_timeseries::stationarity::{
    adf_test as rust_adf, kpss_test as rust_kpss, KpssKind as RustKpssKind,
};

use crate::convert::vec_f64_to_numpy;
use crate::error::stats_to_py;

fn extract_f64_1d(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(arr) = x.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(arr.as_slice()?.to_vec());
    }
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (x, "float64"))?;
    let arr: PyReadonlyArray1<'_, f64> = arr.extract()?;
    Ok(arr.as_slice()?.to_vec())
}

fn extract_usize_1d(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (x, "int64"))?;
    let arr: PyReadonlyArray1<'_, i64> = arr.extract()?;
    Ok(arr.as_slice()?.iter().map(|&v| v as usize).collect())
}

fn forecast_to_dict<'py>(py: Python<'py>, f: RustForecast) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("point", vec_f64_to_numpy(py, f.point))?;
    if let Some(fitted) = f.fitted {
        d.set_item("fitted", vec_f64_to_numpy(py, fitted))?;
    } else {
        d.set_item("fitted", py.None())?;
    }
    if let Some(residuals) = f.residuals {
        d.set_item("residuals", vec_f64_to_numpy(py, residuals))?;
    } else {
        d.set_item("residuals", py.None())?;
    }
    Ok(d)
}

fn parse_ets_model(name: &str) -> PyResult<RustEtsModel> {
    match name.to_ascii_uppercase().as_str() {
        "ANN" => Ok(RustEtsModel::Ann),
        "AAN" => Ok(RustEtsModel::Aan),
        "AAA" => Ok(RustEtsModel::Aaa),
        "MNN" => Ok(RustEtsModel::Mnn),
        _ => Err(PyValueError::new_err(
            "ets model must be one of: ANN, AAN, AAA, MNN",
        )),
    }
}

fn parse_decompose_model(name: &str) -> PyResult<RustDecompose> {
    match name.to_ascii_lowercase().as_str() {
        "additive" => Ok(RustDecompose::Additive),
        "multiplicative" => Ok(RustDecompose::Multiplicative),
        _ => Err(PyValueError::new_err(
            "decompose model must be 'additive' or 'multiplicative'",
        )),
    }
}

fn parse_kpss_kind(name: &str) -> PyResult<RustKpssKind> {
    match name.to_ascii_lowercase().as_str() {
        "level" => Ok(RustKpssKind::Level),
        "trend" => Ok(RustKpssKind::Trend),
        _ => Err(PyValueError::new_err("kpss kind must be 'level' or 'trend'")),
    }
}

/// Naive (last-value) forecast.
#[pyfunction]
fn naive<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    h: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let x = extract_f64_1d(py, x)?;
    let f = rust_naive(&x, h).map_err(stats_to_py)?;
    forecast_to_dict(py, f)
}

/// Seasonal naive forecast with period `period`.
#[pyfunction]
fn seasonal_naive<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    period: usize,
    h: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let x = extract_f64_1d(py, x)?;
    let f = rust_seasonal_naive(&x, period, h).map_err(stats_to_py)?;
    forecast_to_dict(py, f)
}

/// Drift (linear extrapolate from first to last) forecast.
#[pyfunction]
fn drift<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    h: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let x = extract_f64_1d(py, x)?;
    let f = rust_drift(&x, h).map_err(stats_to_py)?;
    forecast_to_dict(py, f)
}

/// Fitted ETS model (`ANN` / `AAN` / `AAA` / `MNN`).
#[pyclass(module = "statscore.timeseries", skip_from_py_object, name = "EtsFit")]
pub struct EtsFit {
    inner: RustEts,
}

#[pymethods]
impl EtsFit {
    /// Fit an ETS model. `period` is required for `AAA`.
    #[classmethod]
    #[pyo3(signature = (x, model, period=None))]
    fn fit(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        model: &str,
        period: Option<usize>,
    ) -> PyResult<Self> {
        let x = extract_f64_1d(py, x)?;
        let m = parse_ets_model(model)?;
        let inner = RustEts::fit(&x, m, period).map_err(stats_to_py)?;
        Ok(Self { inner })
    }

    /// Model class name (`ANN`, `AAN`, `AAA`, or `MNN`).
    #[getter]
    fn model(&self) -> &'static str {
        match self.inner.model() {
            RustEtsModel::Ann => "ANN",
            RustEtsModel::Aan => "AAN",
            RustEtsModel::Aaa => "AAA",
            RustEtsModel::Mnn => "MNN",
        }
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta()
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    /// `h`-step ahead forecast dict with `point`, `fitted`, `residuals`.
    fn forecast<'py>(&self, py: Python<'py>, h: usize) -> PyResult<Bound<'py, PyDict>> {
        let f = Forecaster::forecast(&self.inner, h).map_err(stats_to_py)?;
        forecast_to_dict(py, f)
    }

    fn __repr__(&self) -> String {
        format!("EtsFit(model={})", self.model())
    }
}

/// Fitted ARIMA(p, d, q) model.
#[pyclass(module = "statscore.timeseries", skip_from_py_object, name = "ArimaModel")]
pub struct ArimaModel {
    inner: RustArima,
}

#[pymethods]
impl ArimaModel {
    /// Fit ARIMA(p,d,q) via Yule–Walker AR and innovations MA.
    #[classmethod]
    fn fit(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        p: usize,
        d: usize,
        q: usize,
    ) -> PyResult<Self> {
        let x = extract_f64_1d(py, x)?;
        let inner = RustArima::fit(&x, p, d, q).map_err(stats_to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn p(&self) -> usize {
        self.inner.p
    }

    #[getter]
    fn d(&self) -> usize {
        self.inner.d
    }

    #[getter]
    fn q(&self) -> usize {
        self.inner.q
    }

    #[getter]
    fn ar<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_f64_to_numpy(py, self.inner.ar.clone())
    }

    #[getter]
    fn ma<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_f64_to_numpy(py, self.inner.ma.clone())
    }

    #[getter]
    fn intercept(&self) -> f64 {
        self.inner.intercept
    }

    #[getter]
    fn sigma2(&self) -> f64 {
        self.inner.sigma2
    }

    /// Akaike information criterion (Gaussian).
    fn aic(&self) -> f64 {
        self.inner.aic()
    }

    /// Bayesian information criterion.
    fn bic(&self) -> f64 {
        self.inner.bic()
    }

    fn forecast<'py>(&self, py: Python<'py>, h: usize) -> PyResult<Bound<'py, PyDict>> {
        let f = Forecaster::forecast(&self.inner, h).map_err(stats_to_py)?;
        forecast_to_dict(py, f)
    }

    fn __repr__(&self) -> String {
        format!(
            "ArimaModel(p={}, d={}, q={})",
            self.inner.p, self.inner.d, self.inner.q
        )
    }
}

/// Prophet-style additive OLS model (not Facebook/Stan Prophet).
#[pyclass(module = "statscore.timeseries", skip_from_py_object, name = "ProphetStyleModel")]
pub struct ProphetStyleModel {
    inner: RustProphet,
}

#[pymethods]
impl ProphetStyleModel {
    /// Fit piecewise trend + Fourier seasonality by OLS.
    #[classmethod]
    #[pyo3(signature = (t, y, n_changepoints=5, fourier_order=3, period=365.25))]
    fn fit(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        t: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        n_changepoints: usize,
        fourier_order: usize,
        period: f64,
    ) -> PyResult<Self> {
        let t = extract_f64_1d(py, t)?;
        let y = extract_f64_1d(py, y)?;
        let spec = RustProphetSpec {
            n_changepoints,
            fourier_order,
            period,
        };
        let inner = RustProphet::fit(&t, &y, &spec).map_err(stats_to_py)?;
        Ok(Self { inner })
    }

    /// Predict at future time points `t_future`.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        t_future: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let t = extract_f64_1d(py, t_future)?;
        let f = self.inner.predict(&t).map_err(stats_to_py)?;
        forecast_to_dict(py, f)
    }

    fn __repr__(&self) -> String {
        "ProphetStyleModel()".to_string()
    }
}

/// Discrete-time Markov chain from a state sequence.
#[pyclass(module = "statscore.timeseries", skip_from_py_object, name = "MarkovChain")]
pub struct MarkovChain {
    inner: RustMarkov,
}

#[pymethods]
impl MarkovChain {
    /// Estimate transition matrix from integer states in `0..n_states`.
    #[classmethod]
    fn fit(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        states: &Bound<'_, PyAny>,
        n_states: usize,
    ) -> PyResult<Self> {
        let states = extract_usize_1d(py, states)?;
        let inner = RustMarkov::fit(&states, n_states).map_err(stats_to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn n_states(&self) -> usize {
        self.inner.n_states
    }

    /// Flat row-major transition matrix (length `n²`).
    fn transition<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_f64_to_numpy(py, self.inner.transition_flat())
    }

    /// Stationary distribution π.
    fn stationary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let pi = self.inner.stationary().map_err(stats_to_py)?;
        Ok(vec_f64_to_numpy(py, pi))
    }

    /// One-step predictive distribution from `current`.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        current: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let p = self.inner.predict_proba(current).map_err(stats_to_py)?;
        Ok(vec_f64_to_numpy(py, p))
    }

    /// Simulate a path of length `n` starting from `start`.
    fn sample_path<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        n: usize,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let path = self
            .inner
            .sample_path(start, n, &mut rng())
            .map_err(stats_to_py)?;
        let v: Vec<i64> = path.into_iter().map(|s| s as i64).collect();
        Ok(PyArray1::from_vec(py, v))
    }

    fn __repr__(&self) -> String {
        format!("MarkovChain(n_states={})", self.inner.n_states)
    }
}

/// Augmented Dickey–Fuller unit-root test.
#[pyfunction]
#[pyo3(signature = (x, max_lag=None))]
fn adf_test<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    max_lag: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = extract_f64_1d(py, x)?;
    let r = rust_adf(&x, max_lag).map_err(stats_to_py)?;
    let d = PyDict::new(py);
    d.set_item("statistic", r.statistic)?;
    d.set_item("nobs", r.nobs)?;
    d.set_item("lags", r.lags)?;
    Ok(d)
}

/// KPSS stationarity test (`kind` = `"level"` or `"trend"`).
#[pyfunction]
#[pyo3(signature = (x, kind="level"))]
fn kpss_test<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    kind: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let x = extract_f64_1d(py, x)?;
    let k = parse_kpss_kind(kind)?;
    let r = rust_kpss(&x, k).map_err(stats_to_py)?;
    let d = PyDict::new(py);
    d.set_item("statistic", r.statistic)?;
    d.set_item("nobs", r.nobs)?;
    Ok(d)
}

/// Classical seasonal decomposition.
#[pyfunction]
#[pyo3(signature = (x, period, model="additive"))]
fn classical_decompose<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    period: usize,
    model: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let x = extract_f64_1d(py, x)?;
    let m = parse_decompose_model(model)?;
    let dcmp = rust_decompose(&x, period, m).map_err(stats_to_py)?;
    let d = PyDict::new(py);
    d.set_item("trend", vec_f64_to_numpy(py, dcmp.trend))?;
    d.set_item("seasonal", vec_f64_to_numpy(py, dcmp.seasonal))?;
    d.set_item("residual", vec_f64_to_numpy(py, dcmp.residual))?;
    d.set_item("period", dcmp.period)?;
    d.set_item(
        "model",
        match dcmp.model {
            RustDecompose::Additive => "additive",
            RustDecompose::Multiplicative => "multiplicative",
        },
    )?;
    Ok(d)
}

/// Sample ACF for lags `1..=max_lag`.
#[pyfunction]
fn acf<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    max_lag: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = extract_f64_1d(py, x)?;
    let rho = rust_acf(&x, max_lag).map_err(stats_to_py)?;
    Ok(vec_f64_to_numpy(py, rho))
}

/// Sample PACF (Durbin–Levinson) for lags `1..=max_lag`.
#[pyfunction]
fn pacf<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    max_lag: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = extract_f64_1d(py, x)?;
    let phi = rust_pacf(&x, max_lag).map_err(stats_to_py)?;
    Ok(vec_f64_to_numpy(py, phi))
}

/// Register timeseries types and functions on the `timeseries` submodule.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<EtsFit>()?;
    module.add_class::<ArimaModel>()?;
    module.add_class::<ProphetStyleModel>()?;
    module.add_class::<MarkovChain>()?;
    module.add_function(wrap_pyfunction!(naive, module)?)?;
    module.add_function(wrap_pyfunction!(seasonal_naive, module)?)?;
    module.add_function(wrap_pyfunction!(drift, module)?)?;
    module.add_function(wrap_pyfunction!(adf_test, module)?)?;
    module.add_function(wrap_pyfunction!(kpss_test, module)?)?;
    module.add_function(wrap_pyfunction!(classical_decompose, module)?)?;
    module.add_function(wrap_pyfunction!(acf, module)?)?;
    module.add_function(wrap_pyfunction!(pacf, module)?)?;
    Ok(())
}
