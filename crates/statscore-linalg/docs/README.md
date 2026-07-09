# statscore-linalg

Statistical linear algebra for the `statscore` workspace. Pure-Rust via [nalgebra](https://nalgebra.org) — **no system BLAS required**, compiles on Linux, macOS, Windows, and WASM.

## Documentation index

| Guide | Contents |
|-------|----------|
| [Matrix types & constructors](matrix.md) | `DenseMatrix`, `SquareMatrix`, `Vector`, building matrices |
| [Decompositions](decompositions.md) | Cholesky, QR, SVD, symmetric eigen — math + when to use |
| [Solvers](solve.md) | `solve_linear_system`, `solve_least_squares` |
| [Matrix properties](properties.md) | trace, det, rank, κ, pseudoinverse |
| [Error handling](errors.md) | `StatsError` variants and recovery |
| [Statistical examples](examples.md) | OLS, covariance Cholesky, PCA, condition numbers |

## Design principles

1. **Newtypes, not aliases** — `DenseMatrix` is not `type Matrix = DMatrix<f64>`. `SquareMatrix` enforces `n_rows == n_cols` at construction.
2. **`Result` at boundaries** — decompositions and solvers return `statscore_common::Result`. Invalid inputs never panic.
3. **Row-major for interop** — `from_row_slice` / `as_row_slice` match NumPy/ndarray layout. Internal storage is nalgebra column-major.
4. **Thin decompositions** — QR and SVD use economic (thin) factors to save memory on tall matrices.
5. **Clone-on-decompose** — nalgebra consumes matrices during factorization; we clone the inner matrix (documented cost).

## Quick start

```rust
use statscore_linalg::{
    from_row_slice, identity, square_from_row_slice, vector_from_slice,
    cholesky, qr, svd, eigen_symmetric,
    solve_linear_system, solve_least_squares,
    trace, det, rank, condition_number, pinv,
};

// Build matrices (row-major slices)
let a = square_from_row_slice(2, &[4.0, 2.0, 2.0, 3.0]).unwrap();
let b = vector_from_slice(&[1.0, 2.0]);

// Solve Ax = b (Cholesky for SPD, LU fallback)
let x = solve_linear_system(&a, &b).unwrap();

// Least-squares: design matrix 4×2, response length 4
let design = from_row_slice(4, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();
let y = vector_from_slice(&[1.0, 2.0, 2.0, 3.0]);
let beta = solve_least_squares(&design, &y).unwrap();
```

## Module map

```
statscore-linalg/
├── matrix/           DenseMatrix, SquareMatrix, Vector + constructors
├── decompositions/   cholesky, qr, svd, eigen_symmetric
├── solve/            linear systems + least squares
└── properties/       trace, det, rank, condition_number, pinv
```

## Downstream consumers

| Crate | Uses linalg for |
|-------|-----------------|
| `statscore-distributions` | Cholesky of covariance (multivariate normal) |
| `statscore-regression` | QR OLS, normal equations |
| `statscore-multivariate` | SVD for PCA, eigen for covariance |
| `statscore-hypothesis` | Solving linear systems in ANOVA |

## Dependencies

- [`statscore-common`](../../statscore-common/docs/README.md) — `StatsError`, `Result`
- [`nalgebra`](https://nalgebra.org) 0.35 — pure-Rust LA

## Testing

```bash
cargo test -p statscore-linalg
cargo test -p statscore-linalg --doc
```

20 unit tests + 12 doctests. Cholesky round-trip and SVD reconstruction validated to ~1e-10.

## Status

**Complete** (Phase 0). Optional `nalgebra-lapack` BLAS backend deferred.

## Roadmap

- [ ] `from_ndarray` / `to_ndarray` conversion with `statscore-common::Matrix`
- [ ] `is_positive_definite`, `is_symmetric` helpers
- [ ] Cholesky round-trip property test on 10k random PD matrices
- [ ] Optional `nalgebra-lapack` feature flag
