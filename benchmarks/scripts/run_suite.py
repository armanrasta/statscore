#!/usr/bin/env python3
"""Nightly / local benchmark suite → CSV + HTML with 95% CI on speedups.

Requires a **release** build of `statscore` in the active environment.
"""

from __future__ import annotations

import csv
import datetime as dt
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"
PLOTS = ROOT / "benchmarks" / "plots"


def timed_samples(fn: Callable[[], None], repeats: int = 11, warmup: int = 2) -> list[float]:
    for _ in range(warmup):
        fn()
    out: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def bootstrap_ratio_ci(
    a: list[float], b: list[float], *, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Return (median_ratio, lo, hi) for b/a with percentile bootstrap."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    n = min(len(aa), len(bb))
    aa, bb = aa[:n], bb[:n]
    point = float(np.median(bb) / np.median(aa))
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ratios[i] = np.median(bb[idx]) / np.median(aa[idx])
    lo, hi = np.quantile(ratios, [0.025, 0.975])
    return point, float(lo), float(hi)


def fmt_s(s: float) -> str:
    if s < 1e-6:
        return f"{s * 1e9:.1f} ns"
    if s < 1e-3:
        return f"{s * 1e6:.2f} µs"
    if s < 1.0:
        return f"{s * 1e3:.2f} ms"
    return f"{s:.3f} s"


def main() -> None:
    from scipy import stats

    from statscore.distributions import Gamma, Normal
    from statscore.timeseries import ArimaModel, EtsFit, adf_test, classical_decompose

    try:
        from statsmodels.tsa.arima.model import ARIMA as SMArima
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.stattools import adfuller
    except ImportError as e:
        raise SystemExit("pip install statsmodels") from e

    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    day = dt.date.today().isoformat()
    csv_path = RESULTS / f"bench_{day.replace('-', '_')}.csv"
    html_path = PLOTS / "speedup_summary.html"

    rng = np.random.default_rng(0)
    x = np.linspace(-3, 3, 50_000)
    series = (
        10.0
        + 0.05 * np.arange(500)
        + 3.0 * np.sin(2 * np.pi * np.arange(500) / 12)
        + 0.5 * rng.standard_normal(500)
    )

    rows: list[dict[str, object]] = []

    def add(op: str, competitor: str, ss_fn: Callable[[], None], other_fn: Callable[[], None]) -> None:
        ss = timed_samples(ss_fn)
        ot = timed_samples(other_fn)
        ratio, lo, hi = bootstrap_ratio_ci(ss, ot)
        rows.append(
            {
                "date": day,
                "op": op,
                "competitor": competitor,
                "statscore_median_s": statistics.median(ss),
                "other_median_s": statistics.median(ot),
                "speedup": ratio,
                "speedup_ci95_lo": lo,
                "speedup_ci95_hi": hi,
                "repeats": len(ss),
            }
        )
        print(
            f"{op:<36} {fmt_s(statistics.median(ss)):>10}  "
            f"{fmt_s(statistics.median(ot)):>10}  "
            f"{ratio:7.2f}×  [{lo:.2f}, {hi:.2f}]"
        )

    print(f"statscore benchmarks  {day}  (median + 95% CI on speedup)")
    print(f"{'op':<36} {'statscore':>10} {'other':>10}  {'speedup':>8}  CI95")
    print("-" * 90)

    n = Normal(0.0, 1.0)
    sn = stats.norm()
    add("Normal.pdf scalar", "scipy", lambda: n.pdf(0.5), lambda: sn.pdf(0.5))
    add("Normal.pdf array 50k", "scipy", lambda: n.pdf(x), lambda: sn.pdf(x))
    add("Normal.cdf array 50k", "scipy", lambda: n.cdf(x), lambda: sn.cdf(x))

    g = Gamma(2.5, 1.5)
    sg = stats.gamma(2.5, scale=1.5)
    add("Gamma.cdf array 50k", "scipy", lambda: g.cdf(np.abs(x) + 0.1), lambda: sg.cdf(np.abs(x) + 0.1))

    add(
        "ETS ANN fit+fc h=24",
        "statsmodels",
        lambda: EtsFit.fit(series, "ANN").forecast(24),
        lambda: SimpleExpSmoothing(series, initialization_method="heuristic")
        .fit(smoothing_level=0.3, optimized=False)
        .forecast(24),
    )
    add(
        "ARIMA(1,1,1) fit+fc",
        "statsmodels",
        lambda: ArimaModel.fit(series, 1, 1, 1).forecast(24),
        lambda: SMArima(series, order=(1, 1, 1), trend="n").fit().forecast(24),
    )
    add(
        "ADF maxlag=4",
        "statsmodels",
        lambda: adf_test(series, max_lag=4),
        lambda: adfuller(series, maxlag=4, autolag=None),
    )
    add(
        "decompose additive m=12",
        "statsmodels",
        lambda: classical_decompose(series, 12, model="additive"),
        lambda: seasonal_decompose(series, model="additive", period=12),
    )

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Simple HTML summary
    tr = "\n".join(
        f"<tr><td>{r['op']}</td><td>{r['competitor']}</td>"
        f"<td>{fmt_s(float(r['statscore_median_s']))}</td>"
        f"<td>{fmt_s(float(r['other_median_s']))}</td>"
        f"<td>{float(r['speedup']):.2f}×</td>"
        f"<td>[{float(r['speedup_ci95_lo']):.2f}, {float(r['speedup_ci95_hi']):.2f}]</td></tr>"
        for r in rows
    )
    html_path.write_text(
        f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>statscore speedup {day}</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
th {{ background: #f4f4f4; }}
caption {{ text-align: left; margin-bottom: 0.75rem; font-weight: 600; }}
</style></head><body>
<table>
<caption>statscore vs competitors — {day} (median wall time; speedup = other/statscore; 95% bootstrap CI)</caption>
<thead><tr><th>Op</th><th>Competitor</th><th>statscore</th><th>Other</th><th>Speedup</th><th>95% CI</th></tr></thead>
<tbody>
{tr}
</tbody></table>
<p>Release build required. See <code>benchmarks/README.md</code> and GitHub #26 / #27.</p>
</body></html>
"""
    )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
