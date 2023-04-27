# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - ReleaseDate

### Added

- BVH now works with 2d+ dimensions
- BVH now works with f32/f64
- `simd` feature flag, allows for optimizations via explicit SIMD instructions on nightly 

### Changed

- Moved from glam to nalgebra
- Code uppercase acronyms changed to API conventions (BVH -> Bvh)
- Major performance improvements on BVH optimization