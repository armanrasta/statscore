//! # `statscore-python`
//!
//! Python bindings for the `statscore` workspace via PyO3. Bindings ship
//! **in parallel with each Rust crate**. The Python package depends on
//! **NumPy**; methods accept scalars or arrays.
//!
//! ## Modules
//! - [`distributions`] — Normal, Gamma, Binomial, …
//! - [`fuzzy`] — triangular/trapezoidal numbers, fuzzy logic, fuzzy stats
//! - [`timeseries`] — baselines, ETS, ARIMA, Prophet-style, Markov, ADF/KPSS
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for install and usage.
//! See [performance.md](docs/performance.md) for release benchmarks vs SciPy/NumPy.
//!
//! ## Example
//! ```ignore
//! import numpy as np
//! from statscore.distributions import Normal
//! from statscore.fuzzy import TriangularFuzzyNumber
//! from statscore.timeseries import drift, EtsFit
//!
//! dist = Normal(0.0, 1.0)
//! print(dist.cdf(1.96))
//!
//! warm = TriangularFuzzyNumber(18.0, 22.0, 26.0)
//! print(warm.membership(np.linspace(18, 26, 5)))
//!
//! x = np.arange(40.0)
//! print(drift(x, 5)["point"])
//! ```

#![warn(missing_docs)]
#![allow(unsafe_code)] // required by PyO3 extension modules

mod convert;
mod distributions;
mod error;
mod fuzzy;
mod timeseries;

use pyo3::prelude::*;

/// Python module entry point (`import statscore`).
#[pymodule]
fn statscore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    let dist = PyModule::new(m.py(), "distributions")?;
    distributions::register(&dist)?;
    m.add_submodule(&dist)?;
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("statscore.distributions", &dist)?;

    let fuzzy_mod = PyModule::new(m.py(), "fuzzy")?;
    fuzzy::register(&fuzzy_mod)?;
    m.add_submodule(&fuzzy_mod)?;
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("statscore.fuzzy", &fuzzy_mod)?;

    let ts = PyModule::new(m.py(), "timeseries")?;
    timeseries::register(&ts)?;
    m.add_submodule(&ts)?;
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("statscore.timeseries", &ts)?;

    Ok(())
}
