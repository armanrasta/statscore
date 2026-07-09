# Matrix types & constructors

## Types

### `DenseMatrix`

General `m × n` matrix. Wraps `nalgebra::DMatrix<f64>`.

| Method | Returns | Description |
|--------|---------|-------------|
| `nrows()` | `usize` | Row count |
| `ncols()` | `usize` | Column count |
| `get(row, col)` | `f64` | Element at `(row, col)` — **0-indexed** |
| `as_inner()` | `&DMatrix<f64>` | Borrow nalgebra matrix |
| `into_inner()` | `DMatrix<f64>` | Consume wrapper |
| `as_row_slice()` | `Vec<f64>` | Flatten row-major (NumPy order) |

### `SquareMatrix`

`n × n` matrix. Invariant: `nrows == ncols` enforced by `SquareMatrix::from_inner` (returns `Err` if not square). Constructors like `identity` and `square_from_row_slice` always produce valid squares.

| Method | Returns | Description |
|--------|---------|-------------|
| `dim()` | `usize` | Matrix dimension `n` |
| `get(row, col)` | `f64` | Element access |
| `as_dense()` | `DenseMatrix` | View as general matrix |
| `as_row_slice()` | `Vec<f64>` | Row-major flatten |

### `Vector`

Column vector of length `n`. Wraps `nalgebra::DVector<f64>`.

| Method | Returns | Description |
|--------|---------|-------------|
| `len()` | `usize` | Vector length |
| `is_empty()` | `bool` | True if length 0 |
| `get(i)` | `f64` | Element at index `i` |
| `as_slice()` | `Vec<f64>` | Copy to `Vec` |

## Storage layout

**Important:** nalgebra stores matrices **column-major** internally. User-facing constructors accept **row-major** slices (matching NumPy C-order and `statscore-common::Matrix`).

```rust
// Row-major: [a11, a12, a21, a22] → 2×2 matrix
let m = from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
assert_eq!(m.get(0, 0), 1.0);  // row 0, col 0
assert_eq!(m.get(0, 1), 2.0);
assert_eq!(m.get(1, 0), 3.0);
assert_eq!(m.get(1, 1), 4.0);
```

## Constructors

| Function | Signature | Errors |
|----------|-----------|--------|
| `zeros` | `(rows, cols) → DenseMatrix` | — |
| `ones` | `(rows, cols) → DenseMatrix` | — |
| `identity` | `(n) → SquareMatrix` | — |
| `from_row_slice` | `(rows, cols, &[f64]) → Result<DenseMatrix>` | `DimensionMismatch` if `len != rows×cols` |
| `square_from_row_slice` | `(n, &[f64]) → Result<SquareMatrix>` | `DimensionMismatch` if `len != n²` |
| `vector_from_slice` | `(&[f64]) → Vector` | — |
| `column_vector` | `(len, value) → Vector` | — |

## Examples

### Identity and trace

```rust
use statscore_linalg::{identity, trace};

let i = identity(5);
assert!((trace(&i) - 5.0).abs() < 1e-15);
```

### Design matrix for regression

```rust
use statscore_linalg::from_row_slice;

// Intercept + one predictor, 3 observations
// rows: [1, x1], [1, x2], [1, x3]
let x = from_row_slice(3, 2, &[
    1.0, 2.0,
    1.0, 4.0,
    1.0, 6.0,
]).unwrap();
```

### Covariance matrix (symmetric PD)

```rust
use statscore_linalg::square_from_row_slice;

let sigma = square_from_row_slice(2, &[
    1.0, 0.5,
    0.5, 2.0,
]).unwrap();
```

## Escape hatch

When you need raw nalgebra access:

```rust
let inner = matrix.as_inner();       // borrow
let owned = matrix.into_inner();     // consume
let back = DenseMatrix::from_inner(owned);
```

`SquareMatrix::from_inner(inner)` validates squareness; `from_inner_unchecked` is `pub(crate)` only.
