# statscore-timeseries

Time series analysis and forecasting for quant / stats workflows.

## Overview

Forecasting baselines, ETS, ARIMA, a **Prophet-style** additive regression
(OLS — not Facebook/Stan Prophet), discrete Markov chains, stationarity tests,
and classical seasonal decomposition. Depends only on `statscore-common` and
`statscore-linalg`.

## Modules

| Module | Contents |
|--------|----------|
| `forecast` | `Forecast`, `Forecaster` |
| `baselines` | Naive, seasonal naive, drift |
| `ets` | ETS `ANN` / `AAN` / `AAA` / `MNN` |
| `arima` | ARIMA(p,d,q) fit / forecast / AIC / BIC |
| `prophet` | Piecewise trend + Fourier seasonality (OLS) |
| `markov` | Discrete-time Markov chains |
| `stationarity` | ADF, KPSS |
| `decomposition` | Classical additive / multiplicative |
| `acf` | ACF, PACF |

## Dependencies

- `statscore-common`
- `statscore-linalg`
- `rand` (Markov simulation)

## Example

```rust
use statscore_timeseries::baselines::drift;
use statscore_timeseries::ets::{EtsFit, EtsModel};
use statscore_timeseries::forecast::Forecaster;

let x: Vec<f64> = (0..40).map(|i| i as f64).collect();
let d = drift(&x, 5).unwrap();
let ets = EtsFit::fit(&x, EtsModel::Aan, None).unwrap();
let _ = (d.point, ets.forecast(5).unwrap());
```

```bash
cargo run -p.statscore-timeseries --example timeseries_basics
```

## Python

`statscore.timeseries` — same APIs via PyO3 / NumPy.

## Status

**Phase 2 — implemented (MVP).** Deferred: full Bayesian Prophet, SARIMA, auto-ETS, R `forecast` AIC goldens.
