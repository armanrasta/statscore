"""Tests for statscore.fuzzy (triangular / trapezoidal + logic + stats)."""

from __future__ import annotations

import numpy as np
import pytest

from statscore.fuzzy import (
    TrapezoidalFuzzyNumber,
    TriangularFuzzyNumber,
    fuzzy_and_min,
    fuzzy_correlation,
    fuzzy_mean,
    fuzzy_not,
    fuzzy_or_max,
    fuzzy_variance,
)


def test_triangular_membership():
    warm = TriangularFuzzyNumber(18.0, 22.0, 26.0)
    assert warm.membership(22.0) == 1.0
    assert warm.membership(20.0) == pytest.approx(0.5)
    assert warm.membership(30.0) == 0.0
    assert warm.a == 18.0 and warm.m == 22.0 and warm.b == 26.0


def test_triangular_membership_array():
    t = TriangularFuzzyNumber(1.0, 2.0, 3.0)
    x = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    y = t.membership(x)
    assert isinstance(y, np.ndarray)
    np.testing.assert_allclose(y, [0.0, 0.5, 1.0, 0.5, 0.0])


def test_triangular_defuzzify():
    t = TriangularFuzzyNumber(1.0, 2.0, 6.0)
    assert t.defuzzify_cog() == pytest.approx(3.0)
    assert t.defuzzify_mom() == 2.0
    assert t.core() == [2.0]
    assert t.support() == (1.0, 6.0)
    assert t.alpha_cut(0.5) == (1.5, 4.0)


def test_triangular_invalid():
    with pytest.raises(ValueError):
        TriangularFuzzyNumber(3.0, 2.0, 1.0)


def test_trapezoidal():
    t = TrapezoidalFuzzyNumber(0.0, 1.0, 3.0, 4.0)
    assert t.membership(2.0) == 1.0
    assert t.membership(0.5) == pytest.approx(0.5)
    assert t.defuzzify_cog() == pytest.approx(2.0)
    assert t.defuzzify_mom() == pytest.approx(2.0)


def test_fuzzy_logic():
    assert fuzzy_and_min(0.7, 0.8) == pytest.approx(0.7)
    assert fuzzy_or_max(0.7, 0.8) == pytest.approx(0.8)
    assert fuzzy_not(0.7) == pytest.approx(0.3)


def test_fuzzy_mean_and_variance():
    data = [
        TriangularFuzzyNumber(1.0, 2.0, 3.0),
        TriangularFuzzyNumber(2.0, 3.0, 4.0),
    ]
    mean = fuzzy_mean(data)
    assert mean.m == pytest.approx(2.5)
    assert fuzzy_variance(data) == pytest.approx(0.25)


def test_fuzzy_correlation():
    x = [
        TriangularFuzzyNumber(1.0, 2.0, 3.0),
        TriangularFuzzyNumber(3.0, 4.0, 5.0),
        TriangularFuzzyNumber(5.0, 6.0, 7.0),
    ]
    y = [
        TriangularFuzzyNumber(2.0, 3.0, 4.0),
        TriangularFuzzyNumber(4.0, 5.0, 6.0),
        TriangularFuzzyNumber(6.0, 7.0, 8.0),
    ]
    assert fuzzy_correlation(x, y) == pytest.approx(1.0)
