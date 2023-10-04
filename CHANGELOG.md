# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

- New iterator to traverse BVH that returns intersected shapes in order by distance from ray origin
- New intersection method of aabb with ray, that returns two distances from ray origin to intesections points
- BVH now works with 2d+ dimensions
- BVH now works with f32/f64
- `simd` feature flag, allows for optimizations via explicit SIMD instructions on nightly 

### Changed

- Moved from glam to nalgebra
- Code uppercase acronyms changed to API conventions (BVH -> Bvh)
- Major performance improvements on BVH optimization
