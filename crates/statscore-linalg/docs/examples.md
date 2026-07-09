# Statistical examples

Worked examples showing how `statscore-linalg` supports downstream statistics crates.

## 1. Ordinary least squares (regression)

**Goal:** Fit `y = X β` in the least-squares sense.

```rust
use statscore_linalg::{
    from_row_slice, vector_from_slice, solve_least_squares, qr,
};

// 5 observations, intercept + 2 predictors
let n = 5;
let p = 3;

// Design matrix X (n × p), row-major
let x = from_row_slice(n, p, &[
    1.0, 1.0, 0.0,
    1.0, 0.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 2.0, 0.0,
    1.0, 0.0, 2.0,
]).unwrap();

let y = vector_from_slice(&[2.0, 1.0, 3.0, 3.0, 2.0]);

// Method 1: direct least-squares
let beta = solve_least_squares(&x, &y).unwrap();

// Method 2: explicit QR (same result)
let qr_decomp = qr(&x).unwrap();
let beta2 = qr_decomp.solve_least_squares(&y).unwrap();

// Residuals: e = y - X β
let residuals = y.as_inner() - x.as_inner() * beta.as_inner();
```

**Used by:** `statscore-regression` OLS module.

---

## 2. Cholesky of covariance (multivariate normal)

**Goal:** Factor Σ = L Lᵀ for sampling and density evaluation.

```rust
use statscore_linalg::{square_from_row_slice, cholesky, det};

let mu = [0.0, 0.0]; // mean vector (handled by distributions crate)
let sigma = square_from_row_slice(2, &[
    1.0, 0.6,
    0.6, 1.0,
]).unwrap();

let chol = cholesky(&sigma).unwrap();
let l = chol.l.as_inner();

// Sample: x = μ + L z,  z ~ N(0, I)
// (random z from statscore-distributions)
// let sample = mu + l * z;

// Log-determinant for MVN log-PDF:
// log det(Σ) = 2 * sum_i log(L_ii)
let log_det: f64 = (0..l.nrows())
    .map(|i| l[(i, i)].ln())
    .sum::<f64>() * 2.0;

// Equivalent: log det(Σ) = log(det(Σ))
let log_det_check = det(&sigma).ln();
```

**Used by:** `statscore-distributions` multivariate normal.

---

## 3. PCA via SVD

**Goal:** Principal components of centered data matrix `X` (n × p).

```rust
use statscore_linalg::{from_row_slice, svd};

// Centered data: 4 observations, 3 features
let x = from_row_slice(4, 3, &[
    -1.5, -0.5,  0.5,
    -0.5,  0.5,  1.5,
     0.5, -1.5, -0.5,
     1.5,  1.5, -1.5,
]).unwrap();

let decomp = svd(&x).unwrap();

// Right singular vectors (columns of V) = principal directions
// Singular values² / (n-1) = explained variance
let n = x.nrows() as f64;
let singular_values = &decomp.singular_values;

let explained_variance: Vec<f64> = singular_values
    .iter()
    .map(|&s| s * s / (n - 1.0))
    .collect();

// Scores: T = X V (project onto PCs)
let v = decomp.v_t.as_inner().transpose();
let scores = x.as_inner() * v;
```

**Used by:** `statscore-multivariate` PCA module.

---

## 4. Ill-conditioning diagnostics

**Goal:** Decide whether OLS is trustworthy.

```rust
use statscore_linalg::{
    from_row_slice, condition_number, rank,
    square_from_row_slice,
};

let x = from_row_slice(100, 5, &/* ... nearly collinear columns ... */).unwrap();

// κ(XᵀX) — build normal-equations matrix from public API
let xtx_data: Vec<f64> = (0..5).flat_map(|i| {
    (0..5).map(move |j| {
        (0..100).map(|k| x.get(k, i) * x.get(k, j)).sum::<f64>()
    })
}).collect();
let xtx = square_from_row_slice(5, &xtx_data).unwrap();

let kappa = condition_number(&xtx).unwrap_or(f64::INFINITY);

if kappa > 1e4 {
    eprintln!("Warning: κ = {kappa:.2e}, consider ridge regression");
}

let r = rank(&x, 1e-10).unwrap();
if r < x.ncols() {
    eprintln!("Warning: rank {r} < {} columns", x.ncols());
}
```

**Used by:** `statscore-regression` diagnostics (VIF, condition index).

---

## 5. Pseudoinverse for underdetermined system

**Goal:** Minimum-norm solution when `m < n`.

```rust
use statscore_linalg::{from_row_slice, vector_from_slice, pinv};

// 2 equations, 4 unknowns (underdetermined)
let a = from_row_slice(2, 4, &[
    1.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 1.0,
]).unwrap();
let b = vector_from_slice(&[1.0, 2.0]);

let a_pinv = pinv(&a, 1e-12).unwrap();
let x = a_pinv.as_inner() * b.as_inner();

// x is the minimum-norm solution among all solutions
```

---

## 6. Symmetric eigen decomposition of correlation matrix

```rust
use statscore_linalg::{square_from_row_slice, eigen_symmetric};

let corr = square_from_row_slice(3, &[
    1.0,  0.5,  0.3,
    0.5,  1.0,  0.2,
    0.3,  0.2,  1.0,
]).unwrap();

let eig = eigen_symmetric(&corr).unwrap();

// Eigenvalues = variance explained along each axis
// Eigenvectors = rotation matrix
for (i, &lambda) in eig.eigenvalues.iter().enumerate() {
    println!("PC{} variance fraction: {:.3}", i + 1, lambda / 3.0);
}
```

**Used by:** `statscore-multivariate` factor analysis, spectral methods.
