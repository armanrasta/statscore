#!/usr/bin/env python3
"""Compare statscore.timeseries vs statsmodels (and NumPy baselines).

Fair-ish ops where APIs overlap. Note: algorithms differ (e.g. our ARIMA is
Yule–Walker/CSS, statsmodels often uses MLE; ETS params are fixed defaults
here vs optimized). Timings still show wall-clock cost of comparable workflows.

Run (release extension):
    maturin develop --release
    pip install statsmodels
    python benches/bench_vs_statsmodels_ts.py
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

import numpy as np

from statscore.timeseries import (
    ArimaModel,
    EtsFit,
    MarkovChain,
    ProphetStyleModel,
    acf,
    adf_test,
    classical_decompose,
    drift,
    kpss_test,
    naive,
    pacf,
    seasonal_naive,
)


def timed(fn: Callable[[], None], repeats: int = 7, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def fmt(s: float) -> str:
    if s < 1e-6:
        return f"{s * 1e9:7.1f} ns"
    if s < 1e-3:
        return f"{s * 1e6:7.2f} µs"
    if s < 1.0:
        return f"{s * 1e3:7.2f} ms"
    return f"{s:7.3f} s"


def row(name: str, ss: float, other: float, other_label: str = "other") -> None:
    speedup = other / ss if ss > 0 else float("inf")
    print(f"{name:<40} {fmt(ss):>12} {fmt(other):>12} {speedup:>8.2f}×")


def make_series(n: int = 500, period: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return (
        10.0
        + 0.05 * t
        + 3.0 * np.sin(2.0 * np.pi * t / period)
        + 0.5 * rng.standard_normal(n)
    )


def main() -> None:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
        from statsmodels.tsa.arima.model import ARIMA as SMArima
        from statsmodels.tsa.stattools import acf as sm_acf, adfuller, kpss as sm_kpss, pacf as sm_pacf
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError as e:
        raise SystemExit(
            "statsmodels required: pip install statsmodels\n" + str(e)
        ) from e

    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    x = make_series(500)
    x_short = make_series(120)
    t = np.arange(len(x_short), dtype=float)
    states = np.random.default_rng(0).integers(0, 3, size=2000)

    print("statscore.timeseries vs statsmodels  (median of 7 runs after 2 warmups)")
    print("speedup > 1 ⇒ statscore faster\n")
    print(f"{'op':<40} {'statscore':>12} {'statsmodels':>12} {'speedup':>9}")
    print("-" * 78)

    # --- Baselines (NumPy one-liners as competitor where SM has no naive API) ---
    print("— baselines (vs NumPy) —")
    h = 24

    def np_naive():
        return np.full(h, x[-1])

    def np_drift():
        slope = (x[-1] - x[0]) / (len(x) - 1)
        return x[-1] + slope * np.arange(1, h + 1)

    ss = timed(lambda: naive(x, h))
    ot = timed(np_naive)
    print(f"{'naive h=24':<40} {fmt(ss):>12} {fmt(ot):>12} {ot/ss:>8.2f}×  (NumPy)")

    ss = timed(lambda: drift(x, h))
    ot = timed(np_drift)
    print(f"{'drift h=24':<40} {fmt(ss):>12} {fmt(ot):>12} {ot/ss:>8.2f}×  (NumPy)")

    ss = timed(lambda: seasonal_naive(x, 12, h))
    ot = timed(lambda: np.tile(x[-12:], (h + 11) // 12)[:h])
    print(f"{'seasonal_naive m=12 h=24':<40} {fmt(ss):>12} {fmt(ot):>12} {ot/ss:>8.2f}×  (NumPy)")

    # --- ETS / Holt-Winters ---
    print("\n— ETS / exponential smoothing —")
    # Fit once outside for forecast-only; also time full fit+forecast

    def ss_ses():
        EtsFit.fit(x, "ANN").forecast(h)

    def sm_ses():
        SimpleExpSmoothing(x, initialization_method="estimated").fit(optimized=True).forecast(h)

    row("SES / ETS ANN fit+forecast", timed(ss_ses), timed(sm_ses))

    def ss_holt():
        EtsFit.fit(x, "AAN").forecast(h)

    def sm_holt():
        ExponentialSmoothing(
            x, trend="add", seasonal=None, initialization_method="estimated"
        ).fit(optimized=True).forecast(h)

    row("Holt / ETS AAN fit+forecast", timed(ss_holt), timed(sm_holt))

    def ss_hw():
        EtsFit.fit(x, "AAA", period=12).forecast(h)

    def sm_hw():
        ExponentialSmoothing(
            x,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        ).fit(optimized=True).forecast(h)

    row("Holt-Winters / ETS AAA fit+fc", timed(ss_hw), timed(sm_hw))

    # Fixed-param SES (fairer: our ANN uses α=0.3; SM with fixed smoothing)
    def ss_ses_fixed():
        EtsFit.fit(x, "ANN").forecast(h)

    def sm_ses_fixed():
        SimpleExpSmoothing(x, initialization_method="heuristic").fit(
            smoothing_level=0.3, optimized=False
        ).forecast(h)

    row("SES α=0.3 fixed (fairer)", timed(ss_ses_fixed), timed(sm_ses_fixed))

    # --- ARIMA ---
    print("\n— ARIMA (ours: Yule–Walker/CSS; SM: MLE) —")

    def ss_arima():
        ArimaModel.fit(x, 1, 1, 1).forecast(h)

    def sm_arima():
        # d>0: no constant trend in statsmodels ARIMA
        SMArima(x, order=(1, 1, 1), trend="n").fit().forecast(h)

    row("ARIMA(1,1,1) fit+forecast", timed(ss_arima), timed(sm_arima))

    def ss_ar1():
        ArimaModel.fit(x, 1, 0, 0).forecast(h)

    def sm_ar1():
        SMArima(x, order=(1, 0, 0), trend="c").fit().forecast(h)

    row("AR(1) fit+forecast", timed(ss_ar1), timed(sm_ar1))

    # --- Stationarity ---
    print("\n— stationarity —")
    row("ADF", timed(lambda: adf_test(x, max_lag=4)), timed(lambda: adfuller(x, maxlag=4, autolag=None)))
    row(
        "KPSS level",
        timed(lambda: kpss_test(x, kind="level")),
        timed(lambda: sm_kpss(x, regression="c", nlags="auto")),
    )

    # --- ACF / PACF ---
    print("\n— ACF / PACF —")
    row("ACF maxlag=40", timed(lambda: acf(x, 40)), timed(lambda: sm_acf(x, nlags=40, fft=True)))
    row("PACF maxlag=20", timed(lambda: pacf(x, 20)), timed(lambda: sm_pacf(x, nlags=20, method="ywm")))

    # --- Decomposition ---
    print("\n— classical decompose —")
    row(
        "decompose additive m=12",
        timed(lambda: classical_decompose(x, 12, model="additive")),
        timed(lambda: seasonal_decompose(x, model="additive", period=12)),
    )

    # --- Prophet-style (no fair SM twin; time absolute + vs polyfit baseline) ---
    print("\n— Prophet-style (absolute + NumPy OLS trend baseline) —")

    def ss_prophet():
        m = ProphetStyleModel.fit(
            t, x_short, n_changepoints=5, fourier_order=3, period=12.0
        )
        m.predict(np.arange(120, 144, dtype=float))

    def np_trend():
        coef = np.polyfit(t, x_short, 1)
        np.polyval(coef, np.arange(120, 144, dtype=float))

    ss = timed(ss_prophet)
    ot = timed(np_trend)
    print(f"{'Prophet-style fit+predict':<40} {fmt(ss):>12} {fmt(ot):>12} {ot/ss:>8.2f}×  (polyfit)")

    # --- Markov (absolute; NumPy count baseline) ---
    print("\n— Markov (vs NumPy count normalize) —")

    def ss_markov():
        mc = MarkovChain.fit(states, 3)
        mc.stationary()
        mc.predict_proba(0)
        mc.sample_path(0, 100)

    def np_markov():
        n = 3
        counts = np.zeros((n, n))
        for a, b in zip(states[:-1], states[1:]):
            counts[a, b] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        P = counts / row_sums
        # crude stationary via power iteration
        pi = np.ones(n) / n
        for _ in range(50):
            pi = pi @ P
        _ = P[0]
        # sample
        s = 0
        path = [s]
        for _ in range(99):
            s = int(np.random.choice(n, p=P[s]))
            path.append(s)

    ss = timed(ss_markov)
    ot = timed(np_markov)
    print(f"{'Markov fit+stat+sample':<40} {fmt(ss):>12} {fmt(ot):>12} {ot/ss:>8.2f}×  (NumPy)")

    print("\nNotes:")
    print("  • ETS rows with optimized=True include SM parameter search; fixed-α row is fairer.")
    print("  • ARIMA: different estimators (YW/CSS vs MLE) — speed ≠ accuracy claim.")
    print("  • Use maturin develop --release; debug builds are much slower.")


if __name__ == "__main__":
    main()
