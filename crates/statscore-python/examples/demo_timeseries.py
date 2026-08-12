#!/usr/bin/env python3
"""Demo: baselines, ETS, ARIMA, Prophet-style, Markov, stationarity via statscore.timeseries."""

from __future__ import annotations

import numpy as np

from statscore.timeseries import (
    ArimaModel,
    EtsFit,
    MarkovChain,
    ProphetStyleModel,
    adf_test,
    classical_decompose,
    drift,
    kpss_test,
    naive,
)


def main() -> None:
    t = np.arange(48.0)
    x = 10.0 + 0.2 * t + 2.0 * np.sin(2.0 * np.pi * t / 12.0)

    print("== Baselines ==")
    print("  naive:", naive(x, 3)["point"])
    print("  drift:", drift(x, 3)["point"])

    print("\n== ETS AAA ==")
    ets = EtsFit.fit(x, "AAA", period=12)
    print(f"  model={ets.model}  alpha={ets.alpha:.3f}")
    print("  forecast:", ets.forecast(3)["point"])

    print("\n== ARIMA(1,1,0) ==")
    arima = ArimaModel.fit(x, 1, 1, 0)
    print(f"  AIC={arima.aic():.3f}  BIC={arima.bic():.3f}")
    print("  forecast:", arima.forecast(3)["point"])

    print("\n== Prophet-style (OLS) ==")
    prophet = ProphetStyleModel.fit(t, x, n_changepoints=3, fourier_order=2, period=12.0)
    print("  predict:", prophet.predict([48.0, 49.0, 50.0])["point"])

    print("\n== Stationarity ==")
    print("  ADF:", adf_test(x, max_lag=2))
    print("  KPSS:", kpss_test(x, kind="level"))

    print("\n== Decomposition ==")
    d = classical_decompose(x, 12)
    print("  seasonal[0:3]:", d["seasonal"][:3])

    print("\n== Markov ==")
    states = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int64)
    mc = MarkovChain.fit(states, 2)
    print("  P[0]:", mc.predict_proba(0))
    print("  sample:", mc.sample_path(0, 8))


if __name__ == "__main__":
    main()
