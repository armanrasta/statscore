//! # `statscore-compute`
//!
//! Optional hardware acceleration backends for `statscore`. Feature-gated so
//! the core library compiles everywhere without CUDA/Metal/wgpu installed.
//!
//! ## Planned features
//! - `cuda` — NVIDIA CUDA via `cudarc`
//! - `metal` — Apple Metal (macOS/iOS)
//! - `wgpu` — cross-platform GPU compute
//!
//! ## Dependencies
//! - [`statscore-common`] — types and errors
//! - `rayon` — CPU parallel fallback
//!
//! ## Guide
//!
//! See the [crate guide](docs/README.md) for planned modules and status.
//!
//! ## Status
//! Scaffold crate — implementation pending (post-1.0).
//!
//! [`statscore-common`]: https://docs.rs/statscore-common

#![warn(missing_docs)]
#![forbid(unsafe_code)]
