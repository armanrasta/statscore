# statscore benchmarks

Public **performance** harness for credibility (see [#26](https://github.com/armanrasta/statscore/issues/26)).

Accuracy / numerical validation against SciPy/R lives in [#14](https://github.com/armanrasta/statscore/issues/14) — different goal.

## Layout

```
benchmarks/
├── scripts/
│   ├── run_suite.py          # nightly entrypoint → CSV + HTML
│   ├── vs_scipy_dist.py      # thin wrapper → python benches
│   ├── vs_statsmodels_ts.py
│   └── vs_skfuzzy.py
├── results/                  # CSV artifacts (CI uploads; optional commit)
└── plots/                    # HTML summary from run_suite.py
```

## Local (release only)

```bash
cd crates/statscore-python
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy scipy statsmodels scikit-fuzzy
maturin develop --release

cd ../..
python benchmarks/scripts/run_suite.py
```

## Methodology (short)

- Warmups: 2; timed repeats: 11 (odd → stable median)
- Primary statistic: **median** wall time
- Speedup ratio `T_other / T_statscore`; **95% CI** via percentile bootstrap on paired ratios (B=2000)
- Always release / optimized wheels — debug builds invalidate claims

Full write-up: [#27](https://github.com/armanrasta/statscore/issues/27).

## Nightly CI

`.github/workflows/benchmarks-nightly.yml` runs `run_suite.py` and uploads:

- `results/bench_YYYY_MM_DD.csv`
- `plots/speedup_summary.html`
