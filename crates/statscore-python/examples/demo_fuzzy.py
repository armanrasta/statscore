#!/usr/bin/env python3
"""Demo: fuzzy sets, logic, and statistics via statscore.fuzzy."""

from __future__ import annotations

import numpy as np

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


def main() -> None:
    print("== TriangularFuzzyNumber (warm ≈ 22°C) ==")
    warm = TriangularFuzzyNumber(18.0, 22.0, 26.0)
    xs = np.linspace(18, 26, 5)
    print(f"  membership({xs}) = {warm.membership(xs)}")
    print(f"  support={warm.support()}  core={warm.core()}")
    print(f"  COG={warm.defuzzify_cog():.3f}  MOM={warm.defuzzify_mom():.3f}")

    print("\n== TrapezoidalFuzzyNumber ==")
    comfy = TrapezoidalFuzzyNumber(19.0, 21.0, 24.0, 26.0)
    print(f"  μ(22.5)={comfy.membership(22.5):.2f}  COG={comfy.defuzzify_cog():.3f}")

    print("\n== Fuzzy logic ==")
    print(f"  AND(min, 0.7, 0.4) = {fuzzy_and_min(0.7, 0.4):.2f}")
    print(f"  OR(max, 0.7, 0.4)  = {fuzzy_or_max(0.7, 0.4):.2f}")
    print(f"  NOT(0.7)           = {fuzzy_not(0.7):.2f}")

    print("\n== Fuzzy statistics ==")
    data = [
        TriangularFuzzyNumber(4.5, 5.0, 5.5),
        TriangularFuzzyNumber(4.8, 5.1, 5.4),
        TriangularFuzzyNumber(4.9, 5.0, 5.1),
    ]
    mean = fuzzy_mean(data)
    print(f"  mean = {mean}  COG={mean.defuzzify_cog():.3f}")
    print(f"  variance = {fuzzy_variance(data):.5f}")

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
    print(f"  correlation = {fuzzy_correlation(x, y):.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
