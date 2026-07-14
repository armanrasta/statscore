"""Basic smoke tests for statscore.distributions (no SciPy required)."""

from __future__ import annotations

import math

import pytest

from statscore.distributions import Binomial, Normal, Poisson, standard_normal


def test_version():
    import statscore

    assert isinstance(statscore.__version__, str)


def test_normal_cdf_ppf():
    n = standard_normal()
    assert abs(n.cdf(0.0) - 0.5) < 1e-12
    assert abs(n.ppf(0.5)) < 1e-12
    # Round-trip
    for p in (0.05, 0.25, 0.75, 0.95):
        assert abs(n.cdf(n.ppf(p)) - p) < 1e-10


def test_normal_pdf_at_zero():
    n = Normal(0.0, 1.0)
    expected = 1.0 / math.sqrt(2.0 * math.pi)
    assert abs(n.pdf(0.0) - expected) < 1e-12


def test_normal_invalid_scale():
    with pytest.raises(ValueError):
        Normal(0.0, -1.0)


def test_poisson_mean():
    p = Poisson(3.5)
    assert p.mean() == 3.5
    assert p.pmf(0) > 0


def test_binomial_pmf():
    b = Binomial(10, 0.5)
    assert abs(b.pmf(5) - 252 / 1024) < 1e-12


def test_rvs_shapes():
    n = Normal(0.0, 1.0)
    samples = n.rvs(10)
    assert len(samples) == 10
    assert all(isinstance(x, float) for x in samples)
