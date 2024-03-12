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
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;
use bvh::ray::Ray;
use nalgebra::{Point3, Vector3};

let origin = Point3::new(0.0,0.0,0.0);
let direction = Vector3::new(1.0,0.0,0.0);
let ray = Ray::new(origin, direction);

struct Sphere {
    position: Point3<f32>,
    radius: f32,
    node_index: usize,
}

impl Bounded<f32, 3> for Sphere {
    fn aabb(&self) -> Aabb<f32, 3> {
        let half_size = Vector3::new(self.radius, self.radius, self.radius);
        let min = self.position - half_size;
        let max = self.position + half_size;
        Aabb::with_bounds(min, max)
    }
}

impl BHShape<f32, 3> for Sphere {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }
    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

let mut spheres = Vec::new();
for i in 0..1000u32 {
    let position = Point3::new(i as f32, i as f32, i as f32);
    let radius = (i % 10) as f32 + 1.0;
    spheres.push(Sphere {
        position: position,
        radius: radius,
        node_index: 0,
    });
}

let bvh = Bvh::build_par(&mut spheres);
let hit_sphere_aabbs = bvh.traverse(&ray, &spheres);
```

## Explicit SIMD

This crate features some manually written SIMD instructions, currently only for the `x86_64` architecture.
While nalgebra provides us with generic SIMD optimization (and it does a great job for the most part) - 
some important functions, such as ray-aabb-intersection have been optimized by hand.

The currently optimized intersections for ray-aabb are:
Type: f32, Dimension: 2,3,4
Type: f64, Dimension: 2,3,4

To enable these optimziations, you must build with the `nightly` toolchain and enable the `simd` flag.

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

All of these benchmarks were taken on a Ryzen 3900x.

All benchmarks unless otherwise noted were captured with the simd feature off.

### Intersection via traversal of the list of triangles

```C
test testbase::bench_intersect_120k_triangles_list                            ... bench:     570,717 ns/iter (+/- 21,689)
test testbase::bench_intersect_sponza_list                                    ... bench:     510,683 ns/iter (+/- 9,525)

// simd enabled
test testbase::bench_intersect_120k_triangles_list                            ... bench:     566,916 ns/iter (+/- 22,024)
test testbase::bench_intersect_sponza_list                                    ... bench:     518,821 ns/iter (+/- 12,191)
```

This is the most naive approach to intersecting a scene with a ray. It defines the baseline.

### Intersection via traversal of the list of triangles with AABB checks

```C
test testbase::bench_intersect_120k_triangles_list_aabb                       ... bench:     687,660 ns/iter (+/- 6,850)
test testbase::bench_intersect_sponza_list_aabb                               ... bench:     381,037 ns/iter (+/- 1,416)

// simd enabled
test testbase::bench_intersect_120k_triangles_list_aabb                       ... bench:     295,810 ns/iter (+/- 3,309)
test testbase::bench_intersect_sponza_list_aabb                               ... bench:     163,738 ns/iter (+/- 1,822)
```

AABB checks are cheap, compared to triangle-intersection algorithms. Therefore, preceeding AABB checks
increase intersection speed by filtering negative results a lot faster.

### Build of a BVH from scratch

```C
test flat_bvh::bench::bench_build_1200_triangles_flat_bvh                     ... bench:     242,357 ns/iter (+/- 1,882)
test flat_bvh::bench::bench_build_12k_triangles_flat_bvh                      ... bench:   3,681,965 ns/iter (+/- 223,716)
test flat_bvh::bench::bench_build_120k_triangles_flat_bvh                     ... bench:  46,415,590 ns/iter (+/- 3,226,404)
test bvh::bvh_impl::bench::bench_build_1200_triangles_bvh                     ... bench:     239,473 ns/iter (+/- 2,456)
test bvh::bvh_impl::bench::bench_build_1200_triangles_bvh_rayon               ... bench:     123,387 ns/iter (+/- 9,427)
test bvh::bvh_impl::bench::bench_build_12k_triangles_bvh                      ... bench:   2,903,150 ns/iter (+/- 54,318)
test bvh::bvh_impl::bench::bench_build_12k_triangles_bvh_rayon                ... bench:   1,073,300 ns/iter (+/- 89,530)
test bvh::bvh_impl::bench::bench_build_120k_triangles_bvh                     ... bench:  37,390,480 ns/iter (+/- 2,789,280)
test bvh::bvh_impl::bench::bench_build_120k_triangles_bvh_rayon               ... bench:   8,935,320 ns/iter (+/- 1,780,231)
test bvh::bvh_impl::bench::bench_build_sponza_bvh                             ... bench:  23,828,070 ns/iter (+/- 1,307,461)
test bvh::bvh_impl::bench::bench_build_sponza_bvh_rayon                       ... bench:   4,764,530 ns/iter (+/- 950,640)
```

### Flatten a BVH

```C
test flat_bvh::bench::bench_flatten_120k_triangles_bvh                        ... bench:   9,806,060 ns/iter (+/- 1,771,861)
```

As you can see, building a BVH takes a long time. Building a BVH is only useful if the number of intersections performed on the
scene exceeds the build duration. This is the case in applications such as ray and path tracing, where the minimum
number of intersections is `1280 * 720` for an HD image.

### Intersection via BVH traversal

```C
test flat_bvh::bench::bench_intersect_1200_triangles_flat_bvh                 ... bench:         144 ns/iter (+/- 1)
test flat_bvh::bench::bench_intersect_12k_triangles_flat_bvh                  ... bench:         370 ns/iter (+/- 4)
test flat_bvh::bench::bench_intersect_120k_triangles_flat_bvh                 ... bench:         866 ns/iter (+/- 16)
test bvh::bvh_impl::bench::bench_intersect_1200_triangles_bvh                 ... bench:         146 ns/iter (+/- 2)
test bvh::bvh_impl::bench::bench_intersect_12k_triangles_bvh                  ... bench:         367 ns/iter (+/- 5)
test bvh::bvh_impl::bench::bench_intersect_120k_triangles_bvh                 ... bench:         853 ns/iter (+/- 12)
test bvh::bvh_impl::bench::bench_intersect_sponza_bvh                         ... bench:       1,381 ns/iter (+/- 20)
test ray::ray_impl::bench::bench_intersects_aabb                              ... bench:       4,404 ns/iter (+/- 17)

// simd enabled
test bvh::bvh_impl::bench::bench_intersect_1200_triangles_bvh                 ... bench:         121 ns/iter (+/- 2)
test bvh::bvh_impl::bench::bench_intersect_12k_triangles_bvh                  ... bench:         309 ns/iter (+/- 3)
test bvh::bvh_impl::bench::bench_intersect_120k_triangles_bvh                 ... bench:         749 ns/iter (+/- 15)
test bvh::bvh_impl::bench::bench_intersect_sponza_bvh                         ... bench:       1,216 ns/iter (+/- 16)
test ray::ray_impl::bench::bench_intersects_aabb                              ... bench:       2,447 ns/iter (+/- 18)
```

These performance measurements show that traversing a BVH is much faster than traversing a list.

### Optimization

The benchmarks for how long it takes to update the scene also contain a randomization process which takes some time.

```C
test bvh::optimization::bench::bench_update_shapes_bvh_120k_00p               ... bench:   1,057,965 ns/iter (+/- 76,098)
test bvh::optimization::bench::bench_update_shapes_bvh_120k_01p               ... bench:   2,535,682 ns/iter (+/- 80,231)
test bvh::optimization::bench::bench_update_shapes_bvh_120k_10p               ... bench:  18,840,970 ns/iter (+/- 3,432,867)
test bvh::optimization::bench::bench_update_shapes_bvh_120k_50p               ... bench:  76,025,200 ns/iter (+/- 3,899,770)
test bvh::optimization::bench::bench_randomize_120k_50p                       ... bench:   5,361,645 ns/iter (+/- 436,340)
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
```C
test bvh::optimization::bench::bench_intersect_120k_after_update_shapes_00p   ... bench:         855 ns/iter (+/- 15)
test bvh::optimization::bench::bench_intersect_120k_after_update_shapes_01p   ... bench:         921 ns/iter (+/- 13)
test bvh::optimization::bench::bench_intersect_120k_after_update_shapes_10p   ... bench:       2,677 ns/iter (+/- 191)
test bvh::optimization::bench::bench_intersect_120k_after_update_shapes_50p   ... bench:       2,992 ns/iter (+/- 103)

test bvh::optimization::bench::bench_intersect_120k_with_rebuild_00p          ... bench:         852 ns/iter (+/- 13)
test bvh::optimization::bench::bench_intersect_120k_with_rebuild_01p          ... bench:         918 ns/iter (+/- 13)
test bvh::optimization::bench::bench_intersect_120k_with_rebuild_10p          ... bench:       1,920 ns/iter (+/- 266)
test bvh::optimization::bench::bench_intersect_120k_with_rebuild_50p          ... bench:       2,075 ns/iter (+/- 318)
```

*Sponza*
```C
test bvh::optimization::bench::bench_intersect_sponza_after_update_shapes_00p ... bench:       2,023 ns/iter (+/- 52)
test bvh::optimization::bench::bench_intersect_sponza_after_update_shapes_01p ... bench:       3,444 ns/iter (+/- 120)
test bvh::optimization::bench::bench_intersect_sponza_after_update_shapes_10p ... bench:      16,873 ns/iter (+/- 2,123)
test bvh::optimization::bench::bench_intersect_sponza_after_update_shapes_50p ... bench:       9,075 ns/iter (+/- 486)

test bvh::optimization::bench::bench_intersect_sponza_with_rebuild_00p        ... bench:       1,388 ns/iter (+/- 46)
test bvh::optimization::bench::bench_intersect_sponza_with_rebuild_01p        ... bench:       1,529 ns/iter (+/- 102)
test bvh::optimization::bench::bench_intersect_sponza_with_rebuild_10p        ... bench:       1,920 ns/iter (+/- 120)
test bvh::optimization::bench::bench_intersect_sponza_with_rebuild_50p        ... bench:       2,505 ns/iter (+/- 88)
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
