# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## 0.12.0 - 2025-??-??
- Replace hand-written x86_64 SIMD with safe and portable [`wide`](https://crates.io/crates/wide) SIMD. [#158](https://github.com/svenstaro/bvh/pull/158) (thanks @finnbear)

## 0.11.0 - 2025-02-18
- **Breaking change:** BVH traversal now accepts a `Query: IntersectsAabb` rather than a `Ray`,
  allowing points, AABB's, and circles/spheres to be tested, too. Most use-cases involving `Ray` 
  will continue to compile as-is. If you previously wrote `BvhTraverseIterator<T, D, S>`, you'll
  need to change it to `BvhTraverseIterator<T, D, Ray, S>`. [#128](https://github.com/svenstaro/bvh/pull/128) (thanks @finnbear)
- **Breaking change:** `Ray::intersection_slice_for_aabb` now returns `None` instead of `(-1.0, -1.0)` in the case of no 
  intersection, and `Some((entry, exit))` in the case of intersection. [#133](https://github.com/svenstaro/bvh/pull/133) [#142](https://github.com/svenstaro/bvh/pull/142) (thanks @finnbear)
- `Bvh::nearest_traverse_iterator` and `Bvh::farthest_traverse_iterator` now output correctly ordered results when the children
  of an internal node overlap, resulting in them taking more time and requiring heap allocation.
  The new iterators `Bvh::nearest_child_traverse_iterator` and `Bvh::farthest_child_traverse_iterator` use the old algorithm. [#133](https://github.com/svenstaro/bvh/pull/139) (thanks @dashedman)
- Fix panic on empty `DistanceTraverseIterator` [#117](https://github.com/svenstaro/bvh/pull/117) (thanks @finnbear)
- Fix center() for very large AABBs [#118](https://github.com/svenstaro/bvh/pull/118) (thanks @finnbear)
- Fix more cases where an empty BVH would panic [#116](https://github.com/svenstaro/bvh/pull/116) (thanks @finnbear)
- Add fuzzing suite [#113](https://github.com/svenstaro/bvh/pull/113) (thanks @finnbear)
- Fix some assertions [#129](https://github.com/svenstaro/bvh/pull/129) (thanks @finnbear)
- Fix traversal in case of single-node BVH [#130](https://github.com/svenstaro/bvh/pull/130) (thanks @finnbear)
- Fix intersection between rays and AABBs that lack depth. [#131](https://github.com/svenstaro/bvh/pull/131) (thanks @finnbear)
- Document the limitations of distance traversal best-effort ordering. [#135](https://github.com/svenstaro/bvh/pull/135) (thanks @finnbear)
- Add `Bvh::nearest_to`, which returns the nearest shape to a point. [#108](https://github.com/svenstaro/bvh/pull/108) (thanks @Azkellas)
- Fix ray-AABB intersection such that a ray in the plane of an AABB face is never
  considered intersecting, rather than returning an arbitrary answer in the case of
  `Ray::intersects_aabb` or an erroneous answer in the case of
  `Ray::intersection_slice_for_aabb`. [#149](https://github.com/svenstaro/bvh/pull/149) (thanks @finnbear)

## 0.10.0 - 2024-07-06
- Don't panic when traversing empty BVH [#106](https://github.com/svenstaro/bvh/pull/106) (thanks @finnbear)
- Implement ordered traverse [#98](https://github.com/svenstaro/bvh/pull/98) (thanks @dashedman)

## 0.9.0 - 2024-03-16
- Added an API for allowing the BVH build process to be parallelized and provided an implementation using Rayon under the `rayon` feature flag [#103](https://github.com/svenstaro/bvh/pull/103) (thanks @dbenson24)
- Another round of performance optimizations for the Build operation. Single threaded builds are 4-5x faster and large BVHs with parallelization
are able to build 4-5x faster. There was an almost 15x speedup for building a 120k triangle BVH. [#103](https://github.com/svenstaro/bvh/pull/103) (thanks @dbenson24)
- Trait bounds were consolidated to the BHShape trait instead of being spread across various functions, should have no major implications. [#103](https://github.com/svenstaro/bvh/pull/103) (thanks @dbenson24)

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
