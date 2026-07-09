# statscore-distributions

Probability distributions: continuous, discrete, and multivariate. First user-facing crate.

## Overview

Every distribution implements `ContinuousDistribution` or `DiscreteDistribution` from `statscore-common`. Python bindings ship in the same milestone.

## Planned modules

### Continuous
Normal, Student-t, χ², F, Beta, Gamma, Exponential, Uniform, Logistic, Weibull, Cauchy, Pareto, Gumbel, Laplace, Log-normal, Inverse-gamma, von Mises, Triangular

### Discrete
Binomial, Poisson, Negative binomial, Geometric, Hypergeometric, Multinomial, Discrete uniform

### Multivariate
Multivariate normal, Dirichlet, Wishart, Multivariate t

## Per-distribution API

Each distribution provides: `pdf`/`pmf`, `cdf`, `ppf`, `sample`, moments, stable `log_pdf`/`log_pmf`.

## Dependencies

- `statscore-common`, `statscore-special`, `statscore-linalg`, `statscore-probability`

## Quality gates

- CDF/PPF round-trip to 1e-10
- Validated against SciPy/R
- PyPI `0.1.0-alpha` with Python bindings

## Status

**Scaffold** (Phase 1 MVP).
