//! # `statscore-python`
//!
//! Python bindings for the `statscore` workspace via PyO3. Bindings ship
//! **in parallel with each Rust crate**. The Python package depends on
//! **NumPy**; methods accept scalars or arrays.
//!
//! ## Modules
//! - [`distributions`] — Normal, Gamma, Binomial, …
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for install and usage.
//! See [performance.md](docs/performance.md) for release benchmarks vs SciPy/NumPy.
//!
//! ## Example
//! ```ignore
//! import numpy as np
//! import statscore
//! from statscore.distributions import Normal
//! dist = Normal(0.0, 1.0)
//! print(dist.cdf(1.96))              # float → float
//! print(dist.pdf(np.linspace(-1, 1, 5)))  # ndarray → ndarray
//! ```

#![warn(missing_docs)]
#![allow(unsafe_code)] // required by PyO3 extension modules

mod convert;
mod distributions;
mod error;

use pyo3::prelude::*;

/// Python module entry point (`import statscore`).
#[pymodule]
fn statscore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    let dist = PyModule::new(m.py(), "distributions")?;
    distributions::register(&dist)?;
    m.add_submodule(&dist)?;

    // Allow `from statscore.distributions import Normal` after package install
    // by also registering under sys.modules when used as a flat extension.
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("statscore.distributions", &dist)?;

    Ok(())
}
