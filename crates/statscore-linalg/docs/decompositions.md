# Decompositions

All decompositions clone the input matrix before factorization (nalgebra consumes `self`). Results can be reconstructed to verify accuracy.

## Cholesky — `cholesky`

**Factorization:** `A = L Lᵀ` where `L` is lower-triangular with positive diagonal.

**Requires:** `A` symmetric positive-definite (SPD).

**Returns:** `CholeskyDecomposition { l, … }`

| Method | Description |
|--------|-------------|
| `chol.l` | Lower factor `L` |
| `chol.solve(&b)` | Solve `A x = b` |
| `chol.reconstruct()` | Recover `A = L Lᵀ` |

**Errors:** `NotPositiveDefinite` if Cholesky fails (indefinite or singular matrix).

**Use in statistics:**
- Solve normal equations for SPD systems
- Sample from multivariate normal: `x = μ + L z` where `z ~ N(0, I)`
- Log-determinant: `log det(A) = 2 Σ log Lᵢᵢ`

```rust
use statscore_linalg::{square_from_row_slice, cholesky};

let sigma = square_from_row_slice(2, &[1.0, 0.3, 0.3, 1.0]).unwrap();
let chol = cholesky(&sigma).unwrap();
let l = &chol.l;
// chol.reconstruct() ≈ sigma to ~1e-10
```

---

## QR — `qr`

**Factorization:** `A = Q R` (thin QR for `m × n` with `m ≥ n`).

- `Q`: `m × min(m, n)` with orthonormal columns
- `R`: `min(m, n) × n` upper-triangular

**Returns:** `QrDecomposition { q, r }`

| Method | Description |
|--------|-------------|
| `decomp.reconstruct()` | `Q R ≈ A` |
| `decomp.solve_least_squares(&b)` | Minimize `‖A x − b‖₂` via `x = R⁻¹ Qᵀ b` |

**Use in statistics:**
- OLS regression: `β̂ = (XᵀX)⁻¹ Xᵀ y` via QR on design matrix `X`
- Numerically stable vs. normal equations when κ(X) is large

```rust
use statscore_linalg::{from_row_slice, qr, vector_from_slice};

let x = from_row_slice(4, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();
let y = vector_from_slice(&[1.0, 2.0, 2.0, 3.0]);
let decomp = qr(&x).unwrap();
let beta = decomp.solve_least_squares(&y).unwrap();
```

---

## SVD — `svd`

**Factorization:** `A = U Σ Vᵀ` (thin/economic SVD).

- `U`: `m × k` left singular vectors
- `Σ`: `k` singular values (descending)
- `Vᵀ`: `k × n` right singular vectors transposed

where `k = min(m, n)`.

**Returns:** `SvdDecomposition { u, singular_values, v_t }`

| Method | Description |
|--------|-------------|
| `decomp.reconstruct()` | `U Σ Vᵀ ≈ A` |
| `decomp.rank(tol)` | Count σᵢ > `tol · σ_max` |

**Use in statistics:**
- PCA: eigenvectors of `XᵀX` from SVD of centered `X`
- Rank-deficient regression (pseudoinverse)
- Condition number: `κ = σ_max / σ_min`

```rust
use statscore_linalg::{from_row_slice, svd};

let a = from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
let decomp = svd(&a).unwrap();
println!("singular values: {:?}", decomp.singular_values);
assert_eq!(decomp.rank(1e-10), 2);
```

---

## Symmetric eigen — `eigen_symmetric`

**Factorization:** `A = Q Λ Qᵀ` for real symmetric `A`.

- `Λ`: eigenvalues (ascending)
- `Q`: eigenvector columns

**Requires:** `A` symmetric (only upper triangle used by nalgebra).

**Returns:** `EigenDecomposition { eigenvalues, eigenvectors }`

| Method | Description |
|--------|-------------|
| `decomp.reconstruct()` | `Q Λ Qᵀ ≈ A` |

**Use in statistics:**
- Spectral decomposition of covariance matrices
- Principal component directions (equivalent to SVD of centered data)

**Errors:** `Domain` if `dim == 0`.

```rust
use statscore_linalg::{square_from_row_slice, eigen_symmetric};

let cov = square_from_row_slice(2, &[2.0, 1.0, 1.0, 2.0]).unwrap();
let eig = eigen_symmetric(&cov).unwrap();
// eig.eigenvalues are ascending; largest variance last
```

---

## Choosing a decomposition

| Problem | Decomposition | Why |
|---------|---------------|-----|
| Solve `Ax=b`, A SPD | Cholesky | Fastest, half the memory of LU |
| Solve `Ax=b`, A general | LU (via `solve_linear_system`) | General square systems |
| OLS `min ‖Xβ − y‖` | QR | Stable for ill-conditioned `X` |
| PCA / rank / κ | SVD | Works on any matrix, reveals rank |
| Covariance spectrum | `eigen_symmetric` | Direct when you have Σ |

## Accuracy

Reconstruction tests target ~1e-9 to 1e-10 relative error for well-conditioned random matrices. Ill-conditioned matrices (κ > 10⁸) may lose digits — check `condition_number` first.
