# Solvers

## `solve_linear_system`

Solve the square system **A x = b**.

```rust
pub fn solve_linear_system(a: &SquareMatrix, b: &Vector) -> Result<Vector>
```

### Algorithm

1. **Try Cholesky** — if `A` is SPD, use `CholeskyDecomposition::solve` (fast).
2. **Fall back to LU** — general square solve via nalgebra.

### Requirements

| Constraint | Error |
|------------|-------|
| `b.len() == a.dim()` | `DimensionMismatch` |
| `A` singular | `SingularMatrix` |

### Example

```rust
use statscore_linalg::{
    identity, vector_from_slice, solve_linear_system,
};

let a = identity(3);
let b = vector_from_slice(&[1.0, 2.0, 3.0]);
let x = solve_linear_system(&a, &b).unwrap();

assert!((x.get(0) - 1.0).abs() < 1e-12);
assert!((x.get(1) - 2.0).abs() < 1e-12);
assert!((x.get(2) - 3.0).abs() < 1e-12);
```

### SPD example

```rust
use statscore_linalg::{
    square_from_row_slice, vector_from_slice, solve_linear_system,
};

// [[4, 2], [2, 3]] x = [1, 2]
let a = square_from_row_slice(2, &[4.0, 2.0, 2.0, 3.0]).unwrap();
let b = vector_from_slice(&[1.0, 2.0]);
let x = solve_linear_system(&a, &b).unwrap();

// Verify: A x ≈ b
let ax = a.as_inner() * x.as_inner();
assert!((ax[0] - 1.0).abs() < 1e-10);
assert!((ax[1] - 2.0).abs() < 1e-10);
```

---

## `solve_least_squares`

Solve the overdetermined least-squares problem:

$$\min_x \|A x - b\|_2$$

```rust
pub fn solve_least_squares(a: &DenseMatrix, b: &Vector) -> Result<Vector>
```

### Algorithm

Thin QR decomposition: `x = R^{-1} Q^T b`.

Delegates to `QrDecomposition::solve_least_squares`.

### Requirements

| Constraint | Error |
|------------|-------|
| `b.len() == a.nrows()` | `DimensionMismatch` |
| `R` rank-deficient | `SingularMatrix` |

### Example — simple regression

Fit `y ≈ β₀ + β₁ x` for four points.

```rust
use statscore_linalg::{
    from_row_slice, vector_from_slice, solve_least_squares,
};

// Design matrix: [1, x] for each observation
let design = from_row_slice(4, 2, &[
    1.0, 0.0,   // y=1 at x=0
    1.0, 1.0,   // y=2 at x=1
    1.0, 2.0,   // y=2 at x=2
    1.0, 3.0,   // y=3 at x=3
]).unwrap();

let y = vector_from_slice(&[1.0, 2.0, 2.0, 3.0]);
let beta = solve_least_squares(&design, &y).unwrap();

// beta[0] ≈ intercept, beta[1] ≈ slope
assert_eq!(beta.len(), 2);
```

### When to use least-squares vs. pseudoinverse

| Situation | Use |
|-----------|-----|
| Full column rank, well-conditioned `A` | `solve_least_squares` (faster) |
| Rank-deficient or underdetermined | `pinv` then `A⁺ b` |
| Need minimum-norm solution | `pinv` |

---

## Verification pattern

Always verify solutions in tests and critical paths:

```rust
// For Ax = b:
let residual = a.as_inner() * x.as_inner() - b.as_inner();
assert!(residual.norm() < 1e-8);

// For least squares:
let residual = a.as_inner() * x.as_inner() - b.as_inner();
// ‖residual‖ should be minimal (not necessarily zero)
```
