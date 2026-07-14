//! Convert between Python scalars, sequences, and NumPy arrays.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::stats_to_py;
use statscore_common::Result as StatsResult;

/// Apply `f: f64 → f64` to a Python float **or** a 1-D NumPy `float64` array
/// (or any sequence convertible to one).
pub fn map_f64<'py, F>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    f: F,
) -> PyResult<Bound<'py, PyAny>>
where
    F: Fn(f64) -> f64,
{
    if let Ok(v) = x.extract::<f64>() {
        return Ok(f(v).into_pyobject(py)?.into_any());
    }

    if let Ok(arr) = x.extract::<PyReadonlyArray1<'_, f64>>() {
        let slice = arr.as_slice()?;
        let out: Vec<f64> = slice.iter().copied().map(&f).collect();
        return Ok(PyArray1::from_vec(py, out).into_any());
    }

    // Fall back: try `numpy.asarray(..., dtype=float64)`.
    if let Ok(out) = asarray_f64_then_map(py, x, &f) {
        return Ok(out);
    }

    Err(PyTypeError::new_err(
        "expected a float, a sequence of floats, or a 1-D numpy.ndarray of float64",
    ))
}

/// Apply fallible `f: f64 → Result<f64>` (e.g. ppf).
pub fn map_f64_result<'py, F>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    f: F,
) -> PyResult<Bound<'py, PyAny>>
where
    F: Fn(f64) -> StatsResult<f64>,
{
    if let Ok(v) = x.extract::<f64>() {
        let y = f(v).map_err(stats_to_py)?;
        return Ok(y.into_pyobject(py)?.into_any());
    }

    if let Ok(arr) = x.extract::<PyReadonlyArray1<'_, f64>>() {
        let slice = arr.as_slice()?;
        let mut out = Vec::with_capacity(slice.len());
        for &xi in slice {
            out.push(f(xi).map_err(stats_to_py)?);
        }
        return Ok(PyArray1::from_vec(py, out).into_any());
    }

    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (x, "float64"))?;
    let arr: PyReadonlyArray1<'_, f64> = arr.extract()?;
    let slice = arr.as_slice()?;
    let mut out = Vec::with_capacity(slice.len());
    for &xi in slice {
        out.push(f(xi).map_err(stats_to_py)?);
    }
    Ok(PyArray1::from_vec(py, out).into_any())
}

/// Apply `f: i64 → f64` for discrete pmf/cdf (scalar int or int64 ndarray).
pub fn map_i64_to_f64<'py, F>(
    py: Python<'py>,
    k: &Bound<'py, PyAny>,
    f: F,
) -> PyResult<Bound<'py, PyAny>>
where
    F: Fn(i64) -> f64,
{
    if let Ok(v) = k.extract::<i64>() {
        return Ok(f(v).into_pyobject(py)?.into_any());
    }

    if let Ok(arr) = k.extract::<PyReadonlyArray1<'_, i64>>() {
        let slice = arr.as_slice()?;
        let out: Vec<f64> = slice.iter().copied().map(&f).collect();
        return Ok(PyArray1::from_vec(py, out).into_any());
    }

    // Accept float arrays that are whole numbers (SciPy often passes int via float).
    if let Ok(arr) = k.extract::<PyReadonlyArray1<'_, f64>>() {
        let slice = arr.as_slice()?;
        let out: Vec<f64> = slice
            .iter()
            .map(|&x| f(x as i64))
            .collect();
        return Ok(PyArray1::from_vec(py, out).into_any());
    }

    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (k, "int64"))?;
    let arr: PyReadonlyArray1<'_, i64> = arr.extract()?;
    let slice = arr.as_slice()?;
    let out: Vec<f64> = slice.iter().copied().map(&f).collect();
    Ok(PyArray1::from_vec(py, out).into_any())
}

/// Apply fallible `f: f64 → Result<i64>` (discrete ppf).
pub fn map_f64_to_i64_result<'py, F>(
    py: Python<'py>,
    p: &Bound<'py, PyAny>,
    f: F,
) -> PyResult<Bound<'py, PyAny>>
where
    F: Fn(f64) -> StatsResult<i64>,
{
    if let Ok(v) = p.extract::<f64>() {
        let y = f(v).map_err(stats_to_py)?;
        return Ok(y.into_pyobject(py)?.into_any());
    }

    if let Ok(arr) = p.extract::<PyReadonlyArray1<'_, f64>>() {
        let slice = arr.as_slice()?;
        let mut out = Vec::with_capacity(slice.len());
        for &pi in slice {
            out.push(f(pi).map_err(stats_to_py)?);
        }
        return Ok(PyArray1::from_vec(py, out).into_any());
    }

    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (p, "float64"))?;
    let arr: PyReadonlyArray1<'_, f64> = arr.extract()?;
    let slice = arr.as_slice()?;
    let mut out = Vec::with_capacity(slice.len());
    for &pi in slice {
        out.push(f(pi).map_err(stats_to_py)?);
    }
    Ok(PyArray1::from_vec(py, out).into_any())
}

/// Build a 1-D float64 NumPy array from a `Vec<f64>`.
pub fn vec_f64_to_numpy<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, v)
}

/// Build a 1-D int64 NumPy array from a `Vec<i64>`.
pub fn vec_i64_to_numpy<'py>(py: Python<'py>, v: Vec<i64>) -> Bound<'py, PyArray1<i64>> {
    PyArray1::from_vec(py, v)
}

fn asarray_f64_then_map<'py, F>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    f: &F,
) -> PyResult<Bound<'py, PyAny>>
where
    F: Fn(f64) -> f64,
{
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (x, "float64"))?;
    // 0-d array → scalar path
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if ndim == 0 {
        let v: f64 = arr.call_method0("item")?.extract()?;
        return Ok(f(v).into_pyobject(py)?.into_any());
    }
    let arr: PyReadonlyArray1<'_, f64> = arr.extract()?;
    let slice = arr.as_slice()?;
    let out: Vec<f64> = slice.iter().copied().map(f).collect();
    Ok(PyArray1::from_vec(py, out).into_any())
}
