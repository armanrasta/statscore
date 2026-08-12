#!/usr/bin/env python3
"""Wrapper: timeseries vs statsmodels."""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(
    str(ROOT / "crates/statscore-python/benches/bench_vs_statsmodels_ts.py"),
    run_name="__main__",
)
