# statscore-compute

Optional hardware acceleration backends. Feature-gated — core library compiles without CUDA/Metal/wgpu.

## Planned features

| Feature | Backend |
|---------|---------|
| `cuda` | NVIDIA CUDA via `cudarc` |
| `metal` | Apple Metal (macOS/iOS) |
| `wgpu` | Cross-platform GPU compute |

## Design

- Off by default; no impact on users who don't opt in
- CPU fallback via `rayon` in all code paths
- Separate from `statscore-linalg` (pure-Rust default)

## Dependencies

- `statscore-common`
- `rayon` (CPU parallel)

## Status

**Scaffold** (post-1.0).
