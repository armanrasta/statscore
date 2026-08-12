# statscore documentation

Guides for the **Rust statistics library** and its **Python bindings**.

## Start here

| Topic | Doc |
|-------|-----|
| Project overview | [Root README](../README.md) |
| Probability distributions (Rust) | [statscore-distributions](../crates/statscore-distributions/docs/README.md) |
| Time series / ARIMA / ETS | [statscore-timeseries](../crates/statscore-timeseries/docs/README.md) |
| Fuzzy sets | [statscore-fuzzy](../crates/statscore-fuzzy/docs/README.md) |
| Python install & API | [statscore-python](../crates/statscore-python/docs/README.md) |
| Performance vs SciPy | [performance.md](../crates/statscore-python/docs/performance.md) |
| Public benchmarks | [benchmarks/](../benchmarks/README.md) |

## Common questions

### Is statscore a SciPy alternative?

For many **distribution** and **forecasting** workflows, yes: Rust-native APIs plus Python methods that accept floats or NumPy arrays. It is not a full SciPy replacement (FFT, optimize, etc. are out of scope).

### Does it work with Python and NumPy?

Yes — build `statscore-python` with maturin. Modules: `statscore.distributions`, `statscore.fuzzy`, `statscore.timeseries`.

### Is there an ARIMA / ETS library in Rust?

[`statscore-timeseries`](../crates/statscore-timeseries/docs/README.md) provides naive/drift baselines, ETS, pragmatic ARIMA, Prophet-style OLS, ADF/KPSS, and classical decomposition.

### How fast is it?

See [benchmarks](../benchmarks/README.md) (median + 95% CI) and [performance notes](../crates/statscore-python/docs/performance.md). Always use a **release** build.
