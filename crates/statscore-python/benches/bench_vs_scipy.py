#!/usr/bin/env python3
"""Compare statscore vs SciPy wall-clock timings for common distribution ops.

Run (from venv with scipy + maturin-built statscore):
    python benches/bench_vs_scipy.py
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

import numpy as np
from scipy import stats

from statscore.distributions import (
    Beta,
    Binomial,
    ChiSquared,
    Exponential,
    Gamma,
    Normal,
    Poisson,
    StudentT,
)


def timed(fn: Callable[[], None], repeats: int = 7, warmup: int = 2) -> float:
    """Return median seconds per call."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def fmt(ns: float) -> str:
    if ns < 1e-6:
        return f"{ns * 1e9:7.1f} ns"
    if ns < 1e-3:
        return f"{ns * 1e6:7.2f} µs"
    if ns < 1.0:
        return f"{ns * 1e3:7.2f} ms"
    return f"{ns:7.3f} s"


def row(name: str, ss: float, sp: float) -> None:
    speedup = sp / ss if ss > 0 else float("inf")
    print(f"{name:<32} {fmt(ss):>12} {fmt(sp):>12} {speedup:>8.2f}×")


def main() -> None:
    print("statscore vs SciPy  (median of 7 runs after 2 warmups)")
    print(f"{'op':<32} {'statscore':>12} {'SciPy':>12} {'speedup':>9}")
    print("-" * 68)

    # Scalar ops
    n = Normal(0.0, 1.0)
    sn = stats.norm(0.0, 1.0)
    row("Normal.pdf(0.5)", timed(lambda: n.pdf(0.5)), timed(lambda: sn.pdf(0.5)))
    row("Normal.cdf(1.96)", timed(lambda: n.cdf(1.96)), timed(lambda: sn.cdf(1.96)))
    row("Normal.ppf(0.975)", timed(lambda: n.ppf(0.975)), timed(lambda: sn.ppf(0.975)))

    g = Gamma(2.5, 1.5)
    sg = stats.gamma(2.5, scale=1.5)
    row("Gamma.cdf(3)", timed(lambda: g.cdf(3.0)), timed(lambda: sg.cdf(3.0)))
    row("Gamma.ppf(0.5)", timed(lambda: g.ppf(0.5)), timed(lambda: sg.ppf(0.5)))

    b = Beta(2.0, 5.0)
    sb = stats.beta(2.0, 5.0)
    row("Beta.cdf(0.3)", timed(lambda: b.cdf(0.3)), timed(lambda: sb.cdf(0.3)))
    row("Beta.ppf(0.5)", timed(lambda: b.ppf(0.5)), timed(lambda: sb.ppf(0.5)))

    t = StudentT(10.0)
    st = stats.t(10.0)
    row("StudentT.cdf(1.5)", timed(lambda: t.cdf(1.5)), timed(lambda: st.cdf(1.5)))
    row("StudentT.ppf(0.95)", timed(lambda: t.ppf(0.95)), timed(lambda: st.ppf(0.95)))

    c = ChiSquared(5.0)
    sc = stats.chi2(5.0)
    row("ChiSquared.ppf(0.95)", timed(lambda: c.ppf(0.95)), timed(lambda: sc.ppf(0.95)))

    e = Exponential(1.5)
    se = stats.expon(scale=1 / 1.5)
    row("Exponential.cdf(1)", timed(lambda: e.cdf(1.0)), timed(lambda: se.cdf(1.0)))

    bn = Binomial(20, 0.3)
    sbn = stats.binom(20, 0.3)
    row("Binomial.pmf(6)", timed(lambda: bn.pmf(6)), timed(lambda: sbn.pmf(6)))
    row("Binomial.cdf(6)", timed(lambda: bn.cdf(6)), timed(lambda: sbn.cdf(6)))

    p = Poisson(4.0)
    sp = stats.poisson(4.0)
    row("Poisson.pmf(4)", timed(lambda: p.pmf(4)), timed(lambda: sp.pmf(4)))
    row("Poisson.cdf(4)", timed(lambda: p.cdf(4)), timed(lambda: sp.cdf(4)))

    # Batch PDF/CDF (Python loops vs SciPy vectorized — SciPy should win here until
    # we expose ndarray/numpy bindings; still useful as a baseline)
    xs = np.linspace(-5.0, 5.0, 10_000)
    xs_list = xs.tolist()

    def ss_pdf_loop() -> None:
        for x in xs_list:
            n.pdf(x)

    def ss_cdf_loop() -> None:
        for x in xs_list:
            n.cdf(x)

    row(
        "Normal.pdf ×10k (Python loop)",
        timed(ss_pdf_loop, repeats=5),
        timed(lambda: sn.pdf(xs), repeats=5),
    )
    row(
        "Normal.cdf ×10k (Python loop)",
        timed(ss_cdf_loop, repeats=5),
        timed(lambda: sn.cdf(xs), repeats=5),
    )

    # Sampling
    N = 100_000
    row(
        f"Normal.rvs({N})",
        timed(lambda: n.rvs(N), repeats=5),
        timed(lambda: sn.rvs(N), repeats=5),
    )
    row(
        f"Poisson.rvs({N})",
        timed(lambda: p.rvs(N), repeats=5),
        timed(lambda: sp.rvs(N), repeats=5),
    )

    print()
    print("Notes:")
    print("  • Scalar ops: pure-Rust via PyO3 vs SciPy (C/Fortran).")
    print("  • Batch ×10k: Python for-loop into Rust vs SciPy NumPy vectorization.")
    print("  • Speedup > 1 means statscore is faster.")


if __name__ == "__main__":
    main()
