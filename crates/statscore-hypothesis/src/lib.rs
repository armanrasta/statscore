//! # `statscore-hypothesis`
//!
//! Hypothesis testing: parametric and non-parametric tests, normality checks,
//! multiple-comparison correction, effect sizes, and power analysis.
//!
//! ## Planned modules
//! - `parametric` — t-tests, ANOVA, F-test
//! - `nonparametric` — Mann–Whitney, Wilcoxon, Kruskal–Wallis, Friedman
//! - `normality` — Shapiro–Wilk, Anderson–Darling, Kolmogorov–Smirnov
//! - `proportions` — χ², Fisher exact
//! - `multiple` — Bonferroni, Holm, Benjamini–Hochberg
//! - `effect_size` — Cohen's d, η², ω²
//! - `power` — analytical power for t, z, χ² tests
//!
//! ## Dependencies
//! - [`statscore-common`] — [`HypothesisTest`], [`TestResult`], [`Alternative`]
//! - [`statscore-special`] — distribution CDFs for p-values
//! - [`statscore-distributions`] — null distributions
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for planned modules and status.
//!
//! ## Status
//! Scaffold crate — implementation pending (Phase 1 MVP).
//!
//! [`HypothesisTest`]: statscore_common::HypothesisTest
//! [`TestResult`]: statscore_common::TestResult
//! [`Alternative`]: statscore_common::Alternative

#![warn(missing_docs)]
#![forbid(unsafe_code)]
