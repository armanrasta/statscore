"""Basic smoke tests for statscore.distributions (NumPy for arrays)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from statscore.distributions import Binomial, Normal, Poisson, standard_normal


def test_version():
    import statscore

    assert isinstance(statscore.__version__, str)


def test_normal_cdf_ppf():
    n = standard_normal()
    assert abs(n.cdf(0.0) - 0.5) < 1e-12
    assert abs(n.ppf(0.5)) < 1e-12
    for p in (0.05, 0.25, 0.75, 0.95):
        assert abs(n.cdf(n.ppf(p)) - p) < 1e-10


def test_normal_pdf_at_zero():
    n = Normal(0.0, 1.0)
    expected = 1.0 / math.sqrt(2.0 * math.pi)
    assert abs(n.pdf(0.0) - expected) < 1e-12


def test_normal_pdf_array():
    n = Normal(0.0, 1.0)
    x = np.array([-1.0, 0.0, 1.0])
    y = n.pdf(x)
    assert isinstance(y, np.ndarray)
    assert y.shape == (3,)
    assert abs(y[1] - 1.0 / math.sqrt(2.0 * math.pi)) < 1e-12
    # Matches scalar path
    assert abs(y[0] - n.pdf(-1.0)) < 1e-14


def test_normal_cdf_list():
    n = Normal(0.0, 1.0)
    y = n.cdf([-2.0, 0.0, 2.0])
    assert isinstance(y, np.ndarray)
    assert abs(y[1] - 0.5) < 1e-12


def test_normal_ppf_array():
    n = Normal(0.0, 1.0)
    p = np.array([0.1, 0.5, 0.9])
    q = n.ppf(p)
    assert q.shape == (3,)
    assert abs(q[1]) < 1e-12
    np.testing.assert_allclose(n.cdf(q), p, atol=1e-10)


def test_normal_invalid_scale():
    with pytest.raises(ValueError):
        Normal(0.0, -1.0)


def test_poisson_mean():
    p = Poisson(3.5)
    assert p.mean() == 3.5
    assert p.pmf(0) > 0


def test_poisson_pmf_array():
    p = Poisson(3.0)
    k = np.arange(5, dtype=np.int64)
    y = p.pmf(k)
    assert y.shape == (5,)
    assert abs(y[0] - p.pmf(0)) < 1e-14


def test_binomial_pmf():
    b = Binomial(10, 0.5)
    assert abs(b.pmf(5) - 252 / 1024) < 1e-12


def test_rvs_shapes():
    n = Normal(0.0, 1.0)
    samples = n.rvs(10)
    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.float64
    assert samples.shape == (10,)

    b = Binomial(10, 0.5)
    draws = b.rvs(8)
    assert isinstance(draws, np.ndarray)
    assert draws.dtype == np.int64
    assert draws.shape == (8,)
