#!/usr/bin/env python3
"""Demo: print PDF/CDF/PPF/samples from statscore distributions."""

from __future__ import annotations

import numpy as np

import statscore
from statscore.distributions import (
    Beta,
    Binomial,
    ChiSquared,
    Exponential,
    Gamma,
    Geometric,
    Normal,
    Poisson,
    StudentT,
    Uniform,
    standard_normal,
)


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    print(f"statscore {statscore.__version__}")

    section("Normal(0, 1) — scalar + NumPy")
    n = standard_normal()
    print(f"  pdf(0)     = {n.pdf(0.0):.12f}")
    print(f"  cdf(1.96)  = {n.cdf(1.96):.12f}")
    print(f"  ppf(0.975) = {n.ppf(0.975):.12f}")
    print(f"  mean/var   = {n.mean()}, {n.var()}")
    x = np.linspace(-2, 2, 5)
    print(f"  pdf(x)     = {n.pdf(x)}")
    print(f"  rvs(5)     = {n.rvs(5)}")

    section("Gamma(2, 2)  [shape, scale]")
    g = Gamma(2.0, 2.0)
    print(f"  mean       = {g.mean()}  (expect 4)")
    print(f"  cdf(4)     = {g.cdf(4.0):.12f}")
    print(f"  ppf(0.5)   = {g.ppf(0.5):.12f}")

    section("Beta(2, 5)")
    b = Beta(2.0, 5.0)
    print(f"  mean       = {b.mean():.12f}")
    print(f"  cdf(0.3)   = {b.cdf(0.3):.12f}")

    section("StudentT(10)")
    t = StudentT(10.0)
    print(f"  cdf(0)     = {t.cdf(0.0)}")
    print(f"  ppf(0.95)  = {t.ppf(0.95):.12f}")

    section("ChiSquared(5)")
    c = ChiSquared(5.0)
    print(f"  mean       = {c.mean()}")
    print(f"  ppf(0.95)  = {c.ppf(0.95):.12f}")

    section("Exponential(1.5)")
    e = Exponential(1.5)
    print(f"  mean       = {e.mean():.12f}")
    print(f"  cdf(1)     = {e.cdf(1.0):.12f}")

    section("Uniform(0, 10)")
    u = Uniform(0.0, 10.0)
    print(f"  pdf(3)     = {u.pdf(3.0)}")
    print(f"  ppf(0.25)  = {u.ppf(0.25)}")

    section("Binomial(n=20, p=0.3)")
    bn = Binomial(20, 0.3)
    print(f"  mean       = {bn.mean()}")
    print(f"  pmf(6)     = {bn.pmf(6):.12f}")
    print(f"  cdf(6)     = {bn.cdf(6):.12f}")
    print(f"  ppf(0.5)   = {bn.ppf(0.5)}")
    print(f"  pmf(0..5)  = {bn.pmf(np.arange(6))}")
    print(f"  rvs(8)     = {bn.rvs(8)}")

    section("Poisson(λ=4)")
    p = Poisson(4.0)
    print(f"  pmf(4)     = {p.pmf(4):.12f}")
    print(f"  cdf(4)     = {p.cdf(4):.12f}")
    print(f"  rvs(8)     = {p.rvs(8)}")

    section("Geometric(p=0.25)")
    geo = Geometric(0.25)
    print(f"  mean       = {geo.mean()}")
    print(f"  pmf(0)     = {geo.pmf(0)}")
    print(f"  rvs(8)     = {geo.rvs(8)}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
