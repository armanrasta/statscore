#!/usr/bin/env python3
"""Compare statscore.fuzzy vs scikit-fuzzy (skfuzzy).

Fair ops where both libraries overlap:
  - triangular / trapezoidal membership over a grid
  - closed-form / discrete centroid defuzzification
  - fuzzy AND/OR/NOT on membership arrays
  - batch stats (fuzzy mean / variance) — skfuzzy has no direct API;
    NumPy CoG baseline used as the Python competitor for that op

Run (release extension + scikit-fuzzy):
    maturin develop --release
    pip install scikit-fuzzy
    python benches/bench_vs_skfuzzy.py
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

import numpy as np
import skfuzzy as fuzz

from statscore.fuzzy import (
    TrapezoidalFuzzyNumber,
    TriangularFuzzyNumber,
    fuzzy_and_min,
    fuzzy_and_product,
    fuzzy_correlation,
    fuzzy_mean,
    fuzzy_not,
    fuzzy_or_max,
    fuzzy_or_sum,
    fuzzy_variance,
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


def row(name: str, ss: float, py: float) -> None:
    speedup = py / ss if ss > 0 else float("inf")
    print(f"{name:<44} {fmt(ss):>12} {fmt(py):>12} {speedup:>8.2f}×")


def main() -> None:
    print("statscore.fuzzy vs scikit-fuzzy  (median of 7 runs after 2 warmups)")
    print(f"{'op':<44} {'statscore':>12} {'skfuzzy':>12} {'speedup':>9}")
    print("-" * 80)
    print("  (speedup > 1 means statscore is faster)\n")

    tri_abc = [1.0, 2.0, 6.0]
    trap_abcd = [0.0, 1.0, 3.0, 4.0]
    t = TriangularFuzzyNumber(*tri_abc)
    trap = TrapezoidalFuzzyNumber(*trap_abcd)

    # --- Scalars ---
    print("— scalars —")
    x0 = 2.5
    row(
        "Tri.membership(2.5)",
        timed(lambda: t.membership(x0)),
        timed(lambda: float(fuzz.trimf(np.array([x0]), tri_abc)[0])),
    )
    row(
        "Trap.membership(2.0)",
        timed(lambda: trap.membership(2.0)),
        timed(lambda: float(fuzz.trapmf(np.array([2.0]), trap_abcd)[0])),
    )
    # Defuzzify: statscore closed form vs skfuzzy discrete centroid on dense grid
    x_fine = np.linspace(1.0, 6.0, 2001)
    mf_fine = fuzz.trimf(x_fine, tri_abc)
    row(
        "defuzzify_cog (tri)",
        timed(lambda: t.defuzzify_cog()),
        timed(lambda: fuzz.defuzz(x_fine, mf_fine, "centroid")),
    )
    row(
        "defuzzify_mom (tri)",
        timed(lambda: t.defuzzify_mom()),
        timed(lambda: fuzz.defuzz(x_fine, mf_fine, "mom")),
    )
    print()

    # --- Array membership ---
    print("— membership over grids —")
    for n in (1_000, 10_000, 100_000):
        xs = np.linspace(0.0, 7.0, n)
        xa = np.linspace(-1.0, 5.0, n)

        def ss_tri() -> None:
            t.membership(xs)

        def sk_tri() -> None:
            fuzz.trimf(xs, tri_abc)

        def ss_trap() -> None:
            trap.membership(xa)

        def sk_trap() -> None:
            fuzz.trapmf(xa, trap_abcd)

        row(f"Tri.membership ×{n}", timed(ss_tri), timed(sk_tri))
        row(f"Trap.membership ×{n}", timed(ss_trap), timed(sk_trap))
    print()

    # --- Fuzzy logic on arrays (elementwise) ---
    print("— fuzzy logic (array degrees) —")
    for n in (10_000, 100_000):
        a = np.linspace(0.0, 1.0, n)
        b = np.linspace(1.0, 0.0, n)

        # statscore exposes scalar ops; map via NumPy vectorize for apples-to-
        # oranges note — fairer: pure NumPy min/product vs skfuzzy.fuzzy_*.
        # skfuzzy.fuzzy_and takes (universe, mf) pairs. For equal universes:
        u = np.arange(n, dtype=float)

        def ss_and_min() -> None:
            np.minimum(a, b)  # same math as fuzzy_and_min elementwise

        def sk_and() -> None:
            fuzz.fuzzy_and(u, a, u, b)

        def ss_or_max() -> None:
            np.maximum(a, b)

        def sk_or() -> None:
            fuzz.fuzzy_or(u, a, u, b)

        def ss_not() -> None:
            1.0 - a

        def sk_not() -> None:
            fuzz.fuzzy_not(a)

        # Also time statscore's scalar FFI once (representative of interactive use)
        row(
            f"AND min ×{n} (NumPy min vs skfuzzy)",
            timed(ss_and_min),
            timed(sk_and),
        )
        row(
            f"OR max ×{n} (NumPy max vs skfuzzy)",
            timed(ss_or_max),
            timed(sk_or),
        )
        row(
            f"NOT ×{n} (NumPy vs skfuzzy)",
            timed(ss_not),
            timed(sk_not),
        )
        row(
            f"fuzzy_and_min scalar FFI",
            timed(lambda: fuzzy_and_min(0.7, 0.4)),
            timed(lambda: float(np.minimum(0.7, 0.4))),
        )
    print()

    # Direct statscore scalar logic vs skfuzzy on length-1 arrays
    print("— fuzzy logic scalars — statscore FFI vs skfuzzy(len=1) —")
    u1 = np.array([0.0])
    a1 = np.array([0.7])
    b1 = np.array([0.4])
    row(
        "fuzzy_and_min(0.7, 0.4)",
        timed(lambda: fuzzy_and_min(0.7, 0.4)),
        timed(lambda: float(fuzz.fuzzy_and(u1, a1, u1, b1)[1][0])),
    )
    row(
        "fuzzy_or_max(0.7, 0.4)",
        timed(lambda: fuzzy_or_max(0.7, 0.4)),
        timed(lambda: float(fuzz.fuzzy_or(u1, a1, u1, b1)[1][0])),
    )
    row(
        "fuzzy_not(0.7)",
        timed(lambda: fuzzy_not(0.7)),
        timed(lambda: float(fuzz.fuzzy_not(a1)[0])),
    )
    row(
        "fuzzy_and_product(0.7, 0.4)",
        timed(lambda: fuzzy_and_product(0.7, 0.4)),
        timed(lambda: float(0.7 * 0.4)),  # skfuzzy has no product t-norm helper
    )
    print()

    # --- Fuzzy statistics ---
    print("— fuzzy statistics — statscore vs NumPy-on-COGs (skfuzzy has no fuzzy_mean) —")
    N = 1_000
    rng = np.random.default_rng(0)
    peaks = rng.uniform(2.0, 8.0, size=N)
    data = [
        TriangularFuzzyNumber(float(p - 0.5), float(p), float(p + 0.5)) for p in peaks
    ]
    cogs = np.array([((p - 0.5) + p + (p + 0.5)) / 3.0 for p in peaks])

    def ss_mean() -> None:
        fuzzy_mean(data)

    def np_mean() -> None:
        # Competing "Python" path: mean of CoG of each triangle (same as our var base)
        float(np.mean(cogs))

    def ss_var() -> None:
        fuzzy_variance(data)

    def np_var() -> None:
        float(np.var(cogs))

    row(f"fuzzy_mean (n={N})", timed(ss_mean, repeats=5), timed(np_mean, repeats=5))
    row(f"fuzzy_variance (n={N})", timed(ss_var, repeats=5), timed(np_var, repeats=5))

    # correlation — perfect linear shift
    x = [
        TriangularFuzzyNumber(float(i), float(i + 1), float(i + 2))
        for i in range(200)
    ]
    y = [
        TriangularFuzzyNumber(float(i + 1), float(i + 2), float(i + 3))
        for i in range(200)
    ]
    xd = np.array([(i + (i + 1) + (i + 2)) / 3.0 for i in range(200)])
    yd = np.array([(i + 1 + i + 2 + i + 3) / 3.0 for i in range(200)])

    row(
        "fuzzy_correlation (n=200)",
        timed(lambda: fuzzy_correlation(x, y), repeats=5),
        timed(lambda: float(np.corrcoef(xd, yd)[0, 1]), repeats=5),
    )
    print()

    # Accuracy spot-check
    print("— accuracy spot-check —")
    xs = np.linspace(0.0, 7.0, 10_001)
    ss_mf = np.asarray(t.membership(xs), dtype=float)
    sk_mf = fuzz.trimf(xs, tri_abc)
    print(f"  max |trimf−membership| @10k = {np.max(np.abs(ss_mf - sk_mf)):.3e}")
    cog_ss = t.defuzzify_cog()
    cog_sk = float(fuzz.defuzz(xs, sk_mf, "centroid"))
    print(f"  COG statscore={cog_ss:.6f}  skfuzzy.centroid={cog_sk:.6f}  Δ={abs(cog_ss - cog_sk):.3e}")
    print()
    print("Notes:")
    print("  • Membership: same op — evaluate μ over a grid.")
    print("  • Defuzzify: Closed-form CoG vs discrete centroid on a fixed grid.")
    print("  • Logic arrays: NumPy elementwise = same math as statscore scalar ops;")
    print("    skfuzzy.fuzzy_and/or also align universes (extra work).")
    print("  • Statistics: skfuzzy has no fuzzy_mean/variance; competitor is NumPy")
    print("    on CoGs (fair for variance/correlation; mean differs — we keep vertices).")
    print("  • Build with `maturin develop --release` or numbers look ~10–50× worse.")


if __name__ == "__main__":
    main()
