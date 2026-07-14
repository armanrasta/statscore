# Performance & comparisons

Status of the Python bindings as of mid-2026: **distributions** exposed with
scalar + NumPy array APIs. Numbers below are from a **release** build
(`maturin develop --release`) on Linux x86_64 / CPython 3.14.

> **Always use `--release` for benchmarks and for any speed claim.**  
> A debug extension looked ~10–50× slower than SciPy/NumPy; release reverses that.

## Where we are

| Layer | Status |
|-------|--------|
| Rust `statscore-distributions` | Continuous + discrete families (Normal, Gamma, Beta, Poisson, …) |
| Python `statscore.distributions` | Same set via PyO3 |
| Scalar API | Python `float` / `int` ↔ Rust `f64` / `i64` |
| Array API | 1-D NumPy `ndarray` (or array-like) in one FFI call |
| `rvs` | Always returns `numpy.ndarray` (`float64` / `int64`) |
| Packaging | Depends on `numpy>=1.24`; wheels built by maturin |

We do **not** reimplement NumPy. Arrays are NumPy buffers; kernels run in Rust.

## How array calls work

```
Python ndarray  →  PyO3 reads float64 slice  →  Rust loop (map pdf/cdf/…)
                →  new ndarray allocated as result
```

Scalars pay one FFI cost (~100–300 ns). Large batches amortize that cost.

## Debug vs release (important)

Same machine, `Normal.pdf ×100k` style work:

| Build | vs pure NumPy | vs SciPy vectorized |
|-------|---------------|---------------------|
| **debug** (`maturin develop`) | ~0.2–0.4× (much slower) | CDF/sampling far behind |
| **release** (`maturin develop --release`) | ~1–2× (faster or tied) | PDF/rvs ahead; CDF ≈ parity |

Installed wheels from `maturin build` already use release. Prefer:

```bash
maturin develop --release
```

## Benchmark suite

From `crates/statscore-python` (venv with `numpy`, `scipy`, release extension):

```bash
python benches/bench_statscore_numpy.py   # absolute timings (scalar + arrays)
python benches/bench_vs_scipy.py          # vs SciPy
python benches/bench_vs_numpy.py          # vs hand-rolled NumPy / Generator RNG
```

Rust-only Criterion (no Python):

```bash
cargo bench -p statscore-distributions
```

## Results (release)

Median of 7 runs after 2 warmups. Speedup **> 1 ⇒.statscore faster**.

### Scalars vs SciPy

| Op | statscore | SciPy | Speedup |
|----|-----------|-------|---------|
| `Normal.pdf(0.5)` | 163 ns | 24 µs | **148×** |
| `Normal.cdf(1.96)` | 139 ns | 21 µs | **150×** |
| `Normal.ppf(0.975)` | 230 ns | 29 µs | **124×** |
| `Gamma.cdf(3)` | 144 ns | 20 µs | **141×** |
| `Gamma.ppf(0.5)` | 1.1 µs | 28 µs | **25×** |
| `Beta.cdf(0.3)` | 226 ns | 24 µs | **107×** |
| `Beta.ppf(0.5)` | 1.0 µs | 30 µs | **29×** |
| `StudentT.cdf(1.5)` | 251 ns | 21 µs | **83×** |
| `StudentT.ppf(0.95)` | 1.6 µs | 27 µs | **17×** |
| `ChiSquared.ppf(0.95)` | 1.7 µs | 31 µs | **18×** |
| `Exponential.cdf(1)` | 116 ns | 21 µs | **183×** |
| `Binomial.pmf(6)` | 167 ns | 23 µs | **137×** |
| `Binomial.cdf(6)` | 338 ns | 27 µs | **79×** |
| `Poisson.pmf(4)` | 140 ns | 21 µs | **152×** |
| `Poisson.cdf(4)` | 162 ns | 22 µs | **137×** |

### Batch vs SciPy (10k points / 100k samples)

| Op | statscore | SciPy | Speedup |
|----|-----------|-------|---------|
| `Normal.pdf ×10k` (ndarray) | 42 µs | 118 µs | **2.8×** |
| `Normal.cdf ×10k` (ndarray) | 151 µs | 150 µs | **1.0×** |
| `Normal.rvs(1e5)` | 540 µs | 1.6 ms | **2.9×** |
| `Poisson.rvs(1e5)` | 2.9 ms | 4.6 ms | **1.6×** |

### Batch vs pure NumPy

NumPy column = vectorized formulas (`exp`, A&S `erf`) or `numpy.random.Generator`.

| Op | NumPy | statscore | Speedup |
|----|-------|-----------|---------|
| `Normal.pdf ×1k` | 9 µs | 7 µs | **1.3×** |
| `Normal.cdf ×1k` | 38 µs | 25 µs | **1.5×** |
| `Uniform.pdf ×1k` | 5 µs | 1 µs | **4.7×** |
| `Normal.pdf ×10k` | 59 µs | 64 µs | 0.9× |
| `Normal.cdf ×10k` | 242 µs | 243 µs | **1.0×** |
| `Normal.pdf ×100k` | 945 µs | 479 µs | **2.0×** |
| `Normal.cdf ×100k` | 3.5 ms | 1.6 ms | **2.2×** |
| `Exponential.pdf ×100k` | 588 µs | 308 µs | **1.9×** |
| `Uniform.pdf ×100k` | 75 µs | 30 µs | **2.5×** |
| `Normal.rvs(1e5)` | 973 µs | 575 µs | **1.7×** |
| `Poisson.rvs(1e5)` | 2.9 ms | 2.8 ms | **1.0×** |
| `Normal.rvs(1e6)` | 8.7 ms | 5.3 ms | **1.7×** |
| `Poisson.rvs(1e6)` | 27 ms | 28 ms | 1.0× |

### Absolute NumPy path (statscore only, @100k)

| Op | Total | Per element |
|----|-------|-------------|
| `Uniform.pdf` | 29 µs | **0.3 ns** |
| `Exponential.pdf` | 333 µs | **3.3 ns** |
| `Normal.pdf` | 426 µs | **4.3 ns** |
| `Normal.cdf` | 1.5 ms | **15 ns** |
| `Normal.ppf` | 4.6 ms | **46 ns** |
| `Poisson.pmf` | 2.8 ms | **28 ns** |
| `Binomial.pmf` | 5.0 ms | **50 ns** |
| `Gamma.cdf` | 9.7 ms | **97 ns** |
| `StudentT.cdf` | 18 ms | **179 ns** |
| `Normal.rvs(1e6)` | 5.9 ms | **6 ns** |
| `Poisson.rvs(1e6)` | 29 ms | **29 ns** |

## Takeaways

1. **Scalars:** large wins vs SciPy (typically 80–180×). Ideal for interactive / loop-heavy Python that calls one point at a time.
2. **Arrays:** competitive with SciPy and usually faster than hand-rolled NumPy ufuncs for the same ops once built release.
3. **Sampling:** Normal ahead of SciPy and NumPy RNG; Poisson roughly even.
4. **Debug builds are not comparable** to SciPy/NumPy (those ship optimized C/Fortran).
5. **No custom ndarray library** — keep NumPy as the buffer format; optimize Rust kernels if something falls behind.

## What is still unfinished

- Full SciPy distribution catalog (more continuous/discrete/multivariate).
- Dedicated SIMD / out-of-place write-into-existing-buffer APIs (optional; not required for current parity).
- Automated CI that runs release Python benches and fails on large regressions.
- Publish release notes with these tables when cutting the first PyPI version.

## Related docs

- [Python crate guide](README.md) — install, API table, scalar vs array usage  
- [`statscore-distributions` guide](../../statscore-distributions/docs/README.md) — Rust core  
