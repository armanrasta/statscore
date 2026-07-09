# Matrix properties

## `trace`

Sum of diagonal elements. Only defined for `SquareMatrix`.

```rust
pub fn trace(m: &SquareMatrix) -> f64
```

```rust
use statscore_linalg::{identity, trace};
assert!((trace(&identity(4)) - 4.0).abs() < 1e-15);
```

---

## `det`

Matrix determinant via nalgebra.

```rust
pub fn det(m: &SquareMatrix) -> f64
```

Returns `0.0` for singular matrices (does not error).

```rust
use statscore_linalg::{square_from_row_slice, det};

let m = square_from_row_slice(2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
assert!((det(&m) - (-2.0)).abs() < 1e-12);
```

**Note:** For log-determinants of large PD matrices, prefer Cholesky: `log det(A) = 2 Σ log Lᵢᵢ` (avoids overflow).

---

## `rank`

Numerical rank via SVD.

```rust
pub fn rank(m: &DenseMatrix, tol: f64) -> Result<usize>
```

Counts singular values σᵢ such that **σᵢ / σ_max > tol**.

### Choosing `tol`

| Context | Typical `tol` |
|---------|---------------|
| General | `1e-10` |
| Near machine epsilon | `1e-12` |
| Noisy data | `1e-6` to `1e-8` |

```rust
use statscore_linalg::{from_row_slice, rank};

// Rank-1 matrix: rows are multiples
let m = from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]).unwrap();
assert_eq!(rank(&m, 1e-10).unwrap(), 1);
```

---

## `condition_number`

2-norm condition number κ(A) = σ_max / σ_min.

```rust
pub fn condition_number(m: &SquareMatrix) -> Result<f64>
```

**Errors:** `SingularMatrix` if σ_min ≈ 0 or matrix is empty.

### Interpretation

| κ(A) | Meaning |
|------|---------|
| 1 | Perfectly conditioned (e.g. identity, orthogonal) |
| 10³ | Losing ~3 digits of precision |
| 10⁸ | Ill-conditioned — QR preferred over normal equations |
| ∞ | Singular |

```rust
use statscore_linalg::{identity, condition_number};

assert!((condition_number(&identity(5)).unwrap() - 1.0).abs() < 1e-10);
```

**Use in statistics:** Check κ(X) before OLS. If κ > 10⁴, consider ridge regression or SVD-based solve.

---

## `pinv`

Moore–Penrose pseudoinverse A⁺ via thin SVD.

```rust
pub fn pinv(m: &DenseMatrix, tol: f64) -> Result<DenseMatrix>
```

### Algorithm

1. Compute thin SVD: `A = U Σ Vᵀ`
2. Invert σᵢ only if `σᵢ / σ_max > tol`; else set to 0
3. Return `A⁺ = V Σ⁺ Uᵀ`

### Dimensions

For `A` with shape `m × n`, `A⁺` has shape `n × m`.

### Verification (Penrose conditions)

For full-rank overdetermined `A` (`m > n`):

```
A A⁺ A ≈ A
```

```rust
use statscore_linalg::{from_row_slice, pinv};

let a = from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
let a_pinv = pinv(&a, 1e-12).unwrap();

assert_eq!(a_pinv.nrows(), 2);
assert_eq!(a_pinv.ncols(), 3);

// Penrose: A A⁺ A ≈ A
let check = a.as_inner() * a_pinv.as_inner() * a.as_inner();
// check ≈ a to ~1e-9
```

### Least-squares via pseudoinverse

```rust
let x = a_pinv.as_inner() * b.as_inner();
// x minimizes ‖A x - b‖ when A has full column rank
```

Slower than `solve_least_squares` but works for rank-deficient matrices.
