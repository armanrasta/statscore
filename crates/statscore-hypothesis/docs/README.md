# statscore-hypothesis

Hypothesis testing: parametric and non-parametric tests, normality, multiple comparison correction, effect sizes, and power.

## Planned modules

| Module | Contents |
|--------|----------|
| `parametric` | t-tests, ANOVA, F-test |
| `nonparametric` | Mann–Whitney, Wilcoxon, Kruskal–Wallis, Friedman, runs test |
| `normality` | Shapiro–Wilk, Anderson–Darling, Kolmogorov–Smirnov |
| `proportions` | χ², Fisher exact |
| `correlation` | Tests for r = 0 |
| `multiple` | Bonferroni, Holm, Benjamini–Hochberg |
| `effect_size` | Cohen's d, η², ω² |
| `power` | Analytical power for t, z, χ² |

## Conventions

- `pvalue` field (no underscore) — matches SciPy/R
- Returns `TestResult` from `statscore-common`

## Python bindings

`statscore.hypothesis` — dict/`TestResult` objects matching statsmodels shape.

## Dependencies

- `statscore-common`, `statscore-special`, `statscore-distributions`

## Status

**Scaffold** (Phase 1 MVP).
