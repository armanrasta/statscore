"""Tests for statscore.timeseries."""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.fixture
def trend_seasonal() -> np.ndarray:
    t = np.arange(48.0)
    return 10.0 + 0.2 * t + 2.0 * np.sin(2.0 * np.pi * t / 12.0)


def test_naive_repeats_last():
    f = naive([1.0, 2.0, 9.0], 3)
    np.testing.assert_allclose(f["point"], [9.0, 9.0, 9.0])


def test_seasonal_naive():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    f = seasonal_naive(x, 3, 4)
    np.testing.assert_allclose(f["point"], [4.0, 5.0, 6.0, 4.0])


def test_drift_linear():
    f = drift([1.0, 2.0, 3.0, 4.0], 2)
    np.testing.assert_allclose(f["point"], [5.0, 6.0])


def test_ets_aan(trend_seasonal):
    m = EtsFit.fit(trend_seasonal, "AAN")
    assert m.model == "AAN"
    f = m.forecast(3)
    assert len(f["point"]) == 3
    assert np.all(np.isfinite(f["point"]))


def test_ets_aaa_requires_period(trend_seasonal):
    with pytest.raises(ValueError):
        EtsFit.fit(trend_seasonal, "AAA")
    m = EtsFit.fit(trend_seasonal, "AAA", period=12)
    assert m.model == "AAA"
    assert len(m.forecast(5)["point"]) == 5


def test_arima_forecast_and_ic(trend_seasonal):
    m = ArimaModel.fit(trend_seasonal, 1, 1, 0)
    assert m.p == 1 and m.d == 1 and m.q == 0
    f = m.forecast(3)
    assert len(f["point"]) == 3
    assert np.isfinite(m.aic())
    assert np.isfinite(m.bic())


def test_prophet_style(trend_seasonal):
    t = np.arange(len(trend_seasonal), dtype=float)
    m = ProphetStyleModel.fit(t, trend_seasonal, n_changepoints=3, fourier_order=2, period=12.0)
    f = m.predict([48.0, 49.0, 50.0])
    assert len(f["point"]) == 3


def test_markov():
    states = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int64)
    mc = MarkovChain.fit(states, 2)
    assert mc.n_states == 2
    p = mc.predict_proba(0)
    assert pytest.approx(float(p.sum()), abs=1e-10) == 1.0
    path = mc.sample_path(0, 10)
    assert len(path) == 10
    assert set(path.tolist()).issubset({0, 1})


def test_adf_kpss(trend_seasonal):
    adf = adf_test(trend_seasonal, max_lag=2)
    assert adf["nobs"] > 0
    assert np.isfinite(adf["statistic"])
    kpss = kpss_test(trend_seasonal, kind="level")
    assert kpss["nobs"] == len(trend_seasonal)
    assert np.isfinite(kpss["statistic"])


def test_decompose(trend_seasonal):
    d = classical_decompose(trend_seasonal, 12, model="additive")
    assert d["period"] == 12
    assert len(d["seasonal"]) == len(trend_seasonal)


def test_acf_pacf(trend_seasonal):
    rho = acf(trend_seasonal, 5)
    phi = pacf(trend_seasonal, 5)
    assert len(rho) == 5 and len(phi) == 5
