# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## 0.9.0 - 2024-03-09

- Added an API for allowing the BVH build process to be parallelized and provided an implementation using Rayon under the `rayon` feature flag
- Another round of performance optimizations for the Build operation. Single threaded builds are 4-5x faster and large BVHs with parallelization
are able to build 4-5x faster. There was an almost 15x speedup for building a 120k triangle BVH.
- Trait bounds were consolidated to the BHShape trait instead of being spread across various functions, should have no major implications.


## 0.8.0 - 2024-02-17

- Added ability to incrementally add/remove nodes from tree [#99](https://github.com/svenstaro/bvh/pull/99) (thanks @dbenson24)
- Move math types from glam over to nalgebra with Generic dimensions > 2 and f32/f64 support [#96](https://github.com/svenstaro/bvh/pull/96) (thanks @marstaik)
- BVH now works with 2d+ dimensions
- BVH now works with f32/f64
- `simd` feature flag, allows for optimizations via explicit SIMD instructions on nightly
- Added comments to previously undocumented functions
- Update Rust edition to 2021
- Major performance improvements on BVH optimization
- Code uppercase acronyms changed to API conventions (BVH -> Bvh)
- Fixed all clippy warnings
