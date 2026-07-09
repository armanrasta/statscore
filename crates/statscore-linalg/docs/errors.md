# Error handling

All fallible functions return `statscore_common::Result<T>` (alias for `Result<T, StatsError>`).

## Error variants used by linalg

| Variant | When | Example message |
|---------|------|-----------------|
| `DimensionMismatch` | Incompatible matrix/vector shapes | `"expected rhs length 3, got 2"` |
| `NotPositiveDefinite` | Cholesky fails | `"Cholesky decomposition failed"` |
| `SingularMatrix` | LU/QR/triangular solve fails; κ undefined | `"LU solve failed: matrix is singular"` |
| `Domain` | Invalid parameter (e.g. empty matrix for eigen) | `"matrix dimension must be positive"` |
| `Numerical` | SVD missing expected factors | `"U factor missing"` |

## Non-fallible functions

These return `f64` directly (no `Result`):

| Function | Out-of-domain behavior |
|----------|------------------------|
| `trace` | Always defined for `SquareMatrix` |
| `det` | Returns `0.0` for singular |

## Patterns

### Propagate with `?`

```rust
use statscore_linalg::{from_row_slice, cholesky, solve_linear_system};

fn pipeline(data: &[f64]) -> statscore_common::Result<()> {
    let a = from_row_slice(2, 2, data)?;
    let chol = cholesky(&a)?;
    // ...
    Ok(())
}
```

### Match for recovery

```rust
use statscore_common::StatsError;

match cholesky(&matrix) {
    Ok(chol) => chol.solve(&b),
    Err(StatsError::NotPositiveDefinite(_)) => {
        // Fall back to LU or regularized solve
        solve_linear_system(&matrix, &b)
    }
    Err(e) => Err(e),
}
```

### Python layer mapping (future)

| `StatsError` | Python exception |
|--------------|------------------|
| `DimensionMismatch` | `ValueError` |
| `NotPositiveDefinite` | `ValueError` |
| `SingularMatrix` | `LinAlgError` |
| `Domain` | `ValueError` |
| `Numerical` | `RuntimeError` |

## Avoiding common errors

1. **Check dimensions before solve** — `b.len() == a.dim()` for square; `b.len() == a.nrows()` for least squares.
2. **Verify SPD before Cholesky** — symmetric + positive eigenvalues; or try Cholesky and handle `NotPositiveDefinite`.
3. **Check κ before OLS** — `condition_number` > 10⁴ suggests QR or ridge.
4. **Use consistent row-major** — `from_row_slice` expects C-order data.
