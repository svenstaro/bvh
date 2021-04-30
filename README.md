# bvh
[![CI](https://github.com/svenstaro/bvh/workflows/CI/badge.svg)](https://github.com/svenstaro/bvh/actions)
[![Docs Status](https://docs.rs/bvh/badge.svg)](https://docs.rs/bvh)
[![codecov](https://codecov.io/gh/svenstaro/bvh/branch/master/graph/badge.svg)](https://codecov.io/gh/svenstaro/bvh)
[![license](http://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/svenstaro/bvh/blob/master/LICENSE)
[![Crates.io](https://img.shields.io/crates/v/bvh.svg)](https://crates.io/crates/bvh)
[![Crates.io](https://img.shields.io/crates/d/bvh.svg)](https://crates.io/crates/bvh)

**A crate which exports rays, axis-aligned bounding boxes, and binary bounding
volume hierarchies.**

## About

This crate can be used for applications which contain intersection computations of rays
with primitives. For this purpose a binary tree BVH (Bounding Volume Hierarchy) is of great
use if the scene which the ray traverses contains a huge number of primitives. With a BVH the
intersection test complexity is reduced from O(n) to O(log2(n)) at the cost of building
the BVH once in advance. This technique is especially useful in ray/path tracers. For
use in a shader this module also exports a flattening procedure, which allows for
iterative traversal of the BVH.

## Example

```rust
use bvh::aabb::{AABB, Bounded};
use bvh::bvh::BVH;
use bvh::{Point3, Vector3};
use bvh::ray::Ray;

let origin = Point3::new(0.0,0.0,0.0);
let direction = Vector3::new(1.0,0.0,0.0);
let ray = Ray::new(origin, direction);

struct Sphere {
    position: Point3,
    radius: f32,
}

impl Bounded for Sphere {
    fn aabb(&self) -> AABB {
        let half_size = Vector3::new(self.radius, self.radius, self.radius);
        let min = self.position - half_size;
        let max = self.position + half_size;
        AABB::with_bounds(min, max)
    }
}

let mut spheres = Vec::new();
for i in 0..1000u32 {
    let position = Point3::new(i as f32, i as f32, i as f32);
    let radius = (i % 10) as f32 + 1.0;
    spheres.push(Sphere {
        position: position,
        radius: radius,
    });
}

let bvh = BVH::build(&spheres);
let hit_sphere_aabbs = bvh.traverse_recursive(&ray, &spheres);
```

## Optimization

This crate provides BVH updating, which is also called optimization. With BVH optimization
you can mutate the shapes on which the BVH is built and update the tree accordingly without rebuilding it completely.
This method is very useful when there are only very few changes to a huge scene. When the major part of the scene is static,
it is faster to update the tree, instead of rebuilding it from scratch.

### Drawbacks

First of all, optimizing is not helpful if more than half of the scene is not static.
This is due to how optimizing takes place:
Given a set of indices of all shapes which have changed, the optimize procedure tries to rotate fixed constellations
in search for a better surface area heuristic (SAH) value. This is done recursively from bottom to top while fixing the AABBs
in the inner nodes of the BVH. Which is why it is inefficient to update the BVH in comparison to rebuilding, when a lot
of shapes have moved.

Another problem with updated BVHs is, that the resulting BVH is not optimal. Assume that the scene is composed of two major
groups separated by a large gap. When one shape moves from one group to another, the updating procedure will not be able to
find a sequence of bottom-up rotations which inserts the shape deeply into the other branch.

The following benchmarks are run with two different datasets:
* A randomly generated scene with unit sized cubes containing a total of (1200, 12000, and 120000 triangles).
* Sponza, a popular scene for benchmarking graphics engines.

### Intersection via traversal of the list of triangles

```rust
test testbase::bench_intersect_120k_triangles_list                       ... bench:     653,607 ns/iter (+/- 18,796)
test testbase::bench_intersect_sponza_list                               ... bench:     542,108 ns/iter (+/- 8,705)
```

This is the most naive approach to intersecting a scene with a ray. It defines the baseline.

### Intersection via traversal of the list of triangles with AABB checks

```rust
test testbase::bench_intersect_120k_triangles_list_aabb                  ... bench:     229,088 ns/iter (+/- 6,727)
test testbase::bench_intersect_sponza_list_aabb                          ... bench:     107,514 ns/iter (+/- 1,511)
```

AABB checks are cheap, compared to triangle-intersection algorithms. Therefore, preceeding AABB checks
increase intersection speed by filtering negative results a lot faster.

### Build of a BVH from scratch

```rust
test flat_bvh::bench::bench_build_1200_triangles_flat_bvh                ... bench:     538,474 ns/iter (+/- 4,001)
test flat_bvh::bench::bench_build_12k_triangles_flat_bvh                 ... bench:   6,373,530 ns/iter (+/- 37,217)
test flat_bvh::bench::bench_build_120k_triangles_flat_bvh                ... bench:  74,005,254 ns/iter (+/- 564,271)
test bvh::bvh::bench::bench_build_1200_triangles_bvh                     ... bench:     510,408 ns/iter (+/- 5,240)
test bvh::bvh::bench::bench_build_12k_triangles_bvh                      ... bench:   5,982,294 ns/iter (+/- 31,480)
test bvh::bvh::bench::bench_build_120k_triangles_bvh                     ... bench:  70,182,316 ns/iter (+/- 1,266,142)
test bvh::bvh::bench::bench_build_sponza_bvh                             ... bench:  46,802,305 ns/iter (+/- 184,644)
```

### Flatten a BVH

```rust
test flat_bvh::bench::bench_flatten_120k_triangles_bvh                   ... bench:   3,891,505 ns/iter (+/- 42,360)
```

As you can see, building a BVH takes a long time. Building a BVH is only useful if the number of intersections performed on the
scene exceeds the build duration. This is the case in applications such as ray and path tracing, where the minimum
number of intersections is `1280 * 720` for an HD image.

### Intersection via BVH traversal

```rust
test flat_bvh::bench::bench_intersect_1200_triangles_flat_bvh            ... bench:         168 ns/iter (+/- 2)
test flat_bvh::bench::bench_intersect_12k_triangles_flat_bvh             ... bench:         397 ns/iter (+/- 4)
test flat_bvh::bench::bench_intersect_120k_triangles_flat_bvh            ... bench:         913 ns/iter (+/- 11)
test bvh::bvh::bench::bench_intersect_1200_triangles_bvh                 ... bench:         157 ns/iter (+/- 2)
test bvh::bvh::bench::bench_intersect_12k_triangles_bvh                  ... bench:         384 ns/iter (+/- 6)
test bvh::bvh::bench::bench_intersect_120k_triangles_bvh                 ... bench:         858 ns/iter (+/- 14)
test bvh::bvh::bench::bench_intersect_sponza_bvh                         ... bench:       1,428 ns/iter (+/- 17)
test ray::bench::bench_intersects_aabb                                   ... bench:      34,920 ns/iter (+/- 240)
test ray::bench::bench_intersects_aabb_branchless                        ... bench:      34,867 ns/iter (+/- 214)
test ray::bench::bench_intersects_aabb_naive                             ... bench:      34,958 ns/iter (+/- 259)
```

These performance measurements show that traversing a BVH is much faster than traversing a list.

### Optimization

The benchmarks for how long it takes to update the scene also contain a randomization process which takes some time.

```rust
test bvh::optimization::bench::bench_optimize_bvh_120k_00p               ... bench:   1,123,662 ns/iter (+/- 3,797)
test bvh::optimization::bench::bench_optimize_bvh_120k_01p               ... bench:   6,584,151 ns/iter (+/- 1,375,770)
test bvh::optimization::bench::bench_optimize_bvh_120k_10p               ... bench:  39,725,368 ns/iter (+/- 12,175,627)
test bvh::optimization::bench::bench_optimize_bvh_120k_50p               ... bench: 167,396,675 ns/iter (+/- 55,555,366)
test bvh::optimization::bench::bench_randomize_120k_50p                  ... bench:   3,397,073 ns/iter (+/- 14,335)
```

This is the place where you have to differentiate between rebuilding the tree from scratch or trying to optimize the old one.
These tests show the impact of moving around a particular percentage of shapes (`10p` => `10%`).
It is important to note that the randomization process here moves triangles around indiscriminately.
This will also lead to cases where the BVH would have to be restructured completely.

### Intersection after the optimization

These intersection tests are grouped by dataset and by the BVH generation method.
* `_after_optimize` uses a BVH which was kept up to date with calls to `optimize`, while
* `_with_rebuild` uses the same triangle data as `_after_optimize`, but constructs a BVH from scratch.

*120K Triangles*
```rust
test bvh::optimization::bench::bench_intersect_120k_after_optimize_00p   ... bench:         857 ns/iter (+/- 8)
test bvh::optimization::bench::bench_intersect_120k_after_optimize_01p   ... bench:     139,767 ns/iter (+/- 10,031)
test bvh::optimization::bench::bench_intersect_120k_after_optimize_10p   ... bench:   1,307,082 ns/iter (+/- 315,000)
test bvh::optimization::bench::bench_intersect_120k_after_optimize_50p   ... bench:   2,098,761 ns/iter (+/- 568,199)

test bvh::optimization::bench::bench_intersect_120k_with_rebuild_00p     ... bench:         858 ns/iter (+/- 8)
test bvh::optimization::bench::bench_intersect_120k_with_rebuild_01p     ... bench:         917 ns/iter (+/- 9)
test bvh::optimization::bench::bench_intersect_120k_with_rebuild_10p     ... bench:       1,749 ns/iter (+/- 21)
test bvh::optimization::bench::bench_intersect_120k_with_rebuild_50p     ... bench:       1,988 ns/iter (+/- 35)
```

*Sponza*
```rust
test bvh::optimization::bench::bench_intersect_sponza_after_optimize_00p ... bench:       1,433 ns/iter (+/- 17)
test bvh::optimization::bench::bench_intersect_sponza_after_optimize_01p ... bench:       2,745 ns/iter (+/- 56)
test bvh::optimization::bench::bench_intersect_sponza_after_optimize_10p ... bench:       3,729 ns/iter (+/- 97)
test bvh::optimization::bench::bench_intersect_sponza_after_optimize_50p ... bench:       5,598 ns/iter (+/- 199)

test bvh::optimization::bench::bench_intersect_sponza_with_rebuild_00p   ... bench:       1,426 ns/iter (+/- 45)
test bvh::optimization::bench::bench_intersect_sponza_with_rebuild_01p   ... bench:       1,540 ns/iter (+/- 29)
test bvh::optimization::bench::bench_intersect_sponza_with_rebuild_10p   ... bench:       1,880 ns/iter (+/- 41)
test bvh::optimization::bench::bench_intersect_sponza_with_rebuild_50p   ... bench:       2,375 ns/iter (+/- 54)
```

This set of tests shows the impact of randomly moving triangles around and producing degenerated trees.
The *120K Triangles* dataset has been updated randomly. The *Sponza* scene was updated using a method
which has a maximum offset distance for shapes. This simulates a more realistic scenario.

We also see that the *Sponza* scene by itself contains some structures which can be tightly wrapped in a BVH.
By mowing those structures around we destroy the locality of the triangle groups which leads to more branches in the
BVH requiring a check, thus leading to a higher intersection duration.

### Running the benchmark suite

The benchmark suite uses features from the [test crate](https://doc.rust-lang.org/unstable-book/library-features/test.html) and therefore cannot be run on stable rust.
Using a nightly toolchain, run `cargo bench --features bench`.
