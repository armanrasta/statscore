#!/usr/bin/env python3
"""Compare vectorized NumPy kernels vs statscore NumPy array APIs.

Hand-rolled NumPy formulas / `numpy.random` on the left; one-call statscore
ndarray methods on the right. No SciPy.

Run (venv with numpy + maturin-built statscore):
    python benches/bench_vs_numpy.py
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Callable

import numpy as np

from statscore.distributions import Exponential, Normal, Poisson, Uniform


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


def row(name: str, np_t: float, ss_t: float) -> None:
    # speedup > 1 ⇒ statscore faster than NumPy
    speedup = np_t / ss_t if ss_t > 0 else float("inf")
    print(f"{name:<40} {fmt(np_t):>12} {fmt(ss_t):>12} {speedup:>8.2f}×")


def main() -> None:
    print("NumPy (vectorized / RNG) vs statscore (ndarray APIs)")
    print(f"{'op':<40} {'NumPy':>12} {'statscore':>12} {'speedup':>9}")
    print("-" * 76)
    print("  (speedup > 1 means statscore is faster)\n")

    n = Normal(0.0, 1.0)
    e = Exponential(1.5)
    u = Uniform(0.0, 10.0)
    p = Poisson(4.0)

    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    rate = 1.5
    rng = np.random.default_rng(42)

    # Pure-NumPy erf (Abramowitz & Stegun 7.1.26) — NumPy has no np.erf.
    def erf_np(z: np.ndarray) -> np.ndarray:
        t = 1.0 / (1.0 + 0.5 * np.abs(z))
        tau = t * np.exp(
            -z * z
            - 1.26551223
            + t
            * (
                1.00002368
                + t
                * (
                    0.37409196
                    + t
                    * (
                        0.09678418
                        + t
                        * (
                            -0.18628806
                            + t
                            * (
                                0.27886807
                                + t
                                * (
                                    -1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))
                                )
                            )
                        )
                    )
                )
            )
        )
        return np.where(z >= 0.0, 1.0 - tau, tau - 1.0)

    for size in (1_000, 10_000, 100_000):
        xs = np.linspace(-5.0, 5.0, size)
        xs_pos = np.linspace(0.0, 10.0, size)
        xu = np.linspace(-1.0, 11.0, size)
        print(f"— batch size {size:,} —")

        def np_norm_pdf() -> None:
            _ = inv_sqrt_2pi * np.exp(-0.5 * xs * xs)

        row(f"Normal.pdf ×{size}", timed(np_norm_pdf), timed(lambda: n.pdf(xs)))

        inv_sqrt_2 = 1.0 / math.sqrt(2.0)

        def np_norm_cdf() -> None:
            _ = 0.5 * (1.0 + erf_np(xs * inv_sqrt_2))

        row(f"Normal.cdf ×{size}", timed(np_norm_cdf), timed(lambda: n.cdf(xs)))

        def np_expon_pdf() -> None:
            _ = rate * np.exp(-rate * xs_pos)

        row(
            f"Exponential.pdf ×{size}",
            timed(np_expon_pdf),
            timed(lambda: e.pdf(xs_pos)),
        )

        def np_unif_pdf() -> None:
            _ = np.where((xu >= 0.0) & (xu <= 10.0), 0.1, 0.0)

        row(f"Uniform.pdf ×{size}", timed(np_unif_pdf), timed(lambda: u.pdf(xu)))
        print()

    print("— sampling —")
    for N in (10_000, 100_000, 1_000_000):
        row(
            f"Normal.rvs({N})",
            timed(lambda N=N: rng.normal(0.0, 1.0, size=N), repeats=5),
            timed(lambda N=N: n.rvs(N), repeats=5),
        )
        row(
            f"Poisson.rvs({N})",
            timed(lambda N=N: rng.poisson(4.0, size=N), repeats=5),
            timed(lambda N=N: p.rvs(N), repeats=5),
        )
        print()

    print("Notes:")
    print("  • NumPy = pure ndarray ops / Generator RNG (no SciPy).")
    print("  • Normal.cdf NumPy uses a vectorized A&S erf (no np.erf in NumPy 2).")
    print("  • statscore column = one PyO3 call over a NumPy buffer.")
    print("  • speedup > 1 ⇒ statscore faster.")


if __name__ == "__main__":
    main()
