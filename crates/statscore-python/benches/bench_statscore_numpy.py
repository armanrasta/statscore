#!/usr/bin/env python3
"""Benchmark statscore's NumPy array path (absolute timings).

Measures scalar vs ndarray throughput for distribution methods that accept
both Python numbers and NumPy arrays.

Run (venv with numpy + maturin-built statscore):
    python benches/bench_statscore_numpy.py
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

import numpy as np

from statscore.distributions import (
    Beta,
    Binomial,
    Exponential,
    Gamma,
    Normal,
    Poisson,
    StudentT,
    Uniform,
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


def per_elem(total_s: float, n: int) -> str:
    return fmt(total_s / n)


def row(name: str, total: float, n: int | None = None) -> None:
    if n is None:
        print(f"{name:<42} {fmt(total):>12}")
    else:
        print(f"{name:<42} {fmt(total):>12} {per_elem(total, n):>12}")


def main() -> None:
    print("statscore NumPy path — absolute timings")
    print(f"{'op':<42} {'total':>12} {'per-elem':>12}")
    print("-" * 68)

    n = Normal(0.0, 1.0)
    g = Gamma(2.5, 1.5)
    b = Beta(2.0, 5.0)
    t = StudentT(10.0)
    e = Exponential(1.5)
    u = Uniform(0.0, 10.0)
    bn = Binomial(20, 0.3)
    p = Poisson(4.0)

    print("\n— scalars (Python float / int) —")
    row("Normal.pdf(0.5)", timed(lambda: n.pdf(0.5)))
    row("Normal.cdf(1.96)", timed(lambda: n.cdf(1.96)))
    row("Normal.ppf(0.975)", timed(lambda: n.ppf(0.975)))
    row("Gamma.cdf(3)", timed(lambda: g.cdf(3.0)))
    row("Beta.ppf(0.5)", timed(lambda: b.ppf(0.5)))
    row("StudentT.cdf(1.5)", timed(lambda: t.cdf(1.5)))
    row("Exponential.pdf(1)", timed(lambda: e.pdf(1.0)))
    row("Uniform.pdf(3)", timed(lambda: u.pdf(3.0)))
    row("Binomial.pmf(6)", timed(lambda: bn.pmf(6)))
    row("Poisson.cdf(4)", timed(lambda: p.cdf(4)))

    print("\n— NumPy ndarray (one FFI call) —")
    print(f"{'op':<42} {'total':>12} {'per-elem':>12}")
    print("-" * 68)

    for size in (100, 1_000, 10_000, 100_000):
        xs = np.linspace(-5.0, 5.0, size)
        xs_pos = np.linspace(0.0, 10.0, size)
        ps = np.linspace(0.05, 0.95, size)
        k = np.arange(size, dtype=np.int64) % 21

        print(f"\n  size = {size:,}")
        row(f"Normal.pdf(x)", timed(lambda: n.pdf(xs)), size)
        row(f"Normal.cdf(x)", timed(lambda: n.cdf(xs)), size)
        row(f"Normal.ppf(p)", timed(lambda: n.ppf(ps), repeats=5), size)
        row(f"Gamma.cdf(x)", timed(lambda: g.cdf(xs_pos)), size)
        row(f"Beta.cdf(x)", timed(lambda: b.cdf(np.clip(xs_pos / 10.0, 0, 1))), size)
        row(f"StudentT.cdf(x)", timed(lambda: t.cdf(xs)), size)
        row(f"Exponential.pdf(x)", timed(lambda: e.pdf(xs_pos)), size)
        row(f"Uniform.pdf(x)", timed(lambda: u.pdf(xs_pos)), size)
        row(f"Binomial.pmf(k)", timed(lambda: bn.pmf(k)), size)
        row(f"Poisson.pmf(k)", timed(lambda: p.pmf(k % 30)), size)

    print("\n— rvs → ndarray —")
    print(f"{'op':<42} {'total':>12} {'per-elem':>12}")
    print("-" * 68)
    for size in (1_000, 10_000, 100_000, 1_000_000):
        row(f"Normal.rvs({size})", timed(lambda s=size: n.rvs(s), repeats=5), size)
        row(f"Poisson.rvs({size})", timed(lambda s=size: p.rvs(s), repeats=5), size)

    print()
    print("Notes:")
    print("  • Arrays: one PyO3 call over a float64/int64 NumPy buffer.")
    print("  • per-elem = total / size (includes alloc + FFI).")


if __name__ == "__main__":
    main()
