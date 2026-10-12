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

It uses `rayon` to parallelize building the BVH and `wide` for explicit SIMD.
Both of these are enabled by default.
Make sure to use `RUSTFLAGS='-C target-cpu=native'` for the best possible speedup.
It can also be used in `no_std` contexts by disabling default features.

## Example

```rust
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::{BHShape, BoundingHierarchy};
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

This crate features some SIMD operations. See [`wide`](https://crates.io/crates/wide) documentation for supported architectures.
While nalgebra provides us with generic SIMD optimization (and it does a great job for the most part) -
some important functions, such as ray-aabb-intersection have been optimized by hand.

The currently optimized intersections for ray-aabb are:

- f32, dimensions 2, 3, 4
- f64, dimensions 2, 3, 4

These optimizations are enabled by default via the `simd` feature.

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

All benchmarks were captured with [`divan`](https://crates.io/crates/divan) via `cargo bench` on an **AMD Ryzen 9 9950X3D**, built with `RUSTFLAGS="-C target-cpu=native"`. Numbers are machine-specific snapshots; see "Running the benchmark suite" below to reproduce them.

### Build

Benchmarks for building a BVH from scratch and flattening it.

```text
build                                       fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ build_sponza                             11.42 ms      │ 15.54 ms      │ 11.53 ms      │ 11.84 ms      │ 85      │ 85
├─ build_sponza_par                         1.555 ms      │ 2.879 ms      │ 1.644 ms      │ 1.677 ms      │ 100     │ 100
├─ flatten_120k                             955.5 µs      │ 2.388 ms      │ 1 ms          │ 1.02 ms       │ 100     │ 100
├─ build_triangles                                        │               │               │               │         │
│  ├─ Bvh<f32, 3>                                         │               │               │               │         │
│  │  ├─ 100                                124.5 µs      │ 155.9 µs      │ 125.7 µs      │ 127.4 µs      │ 100     │ 100
│  │  ├─ 1000                               1.514 ms      │ 1.664 ms      │ 1.531 ms      │ 1.532 ms      │ 100     │ 100
│  │  ╰─ 10000                              18.61 ms      │ 19.73 ms      │ 18.72 ms      │ 18.76 ms      │ 54      │ 54
│  ╰─ Vec<bvh::flat_bvh::FlatNode<f32, 3>>                │               │               │               │         │
│     ├─ 100                                132.7 µs      │ 145.5 µs      │ 134.3 µs      │ 134.8 µs      │ 100     │ 100
│     ├─ 1000                               1.602 ms      │ 1.662 ms      │ 1.614 ms      │ 1.617 ms      │ 100     │ 100
│     ╰─ 10000                              21.28 ms      │ 22.22 ms      │ 21.5 ms       │ 21.52 ms      │ 46      │ 46
╰─ build_triangles_par                                    │               │               │               │         │
   ├─ Bvh<f32, 3>                                         │               │               │               │         │
   │  ├─ 100                                42.35 µs      │ 208.8 µs      │ 68.38 µs      │ 78.83 µs      │ 100     │ 100
   │  ├─ 1000                               304.7 µs      │ 729.9 µs      │ 343.5 µs      │ 350.7 µs      │ 100     │ 100
   │  ╰─ 10000                              2.73 ms       │ 3.821 ms      │ 2.916 ms      │ 2.97 ms       │ 100     │ 100
   ╰─ Vec<bvh::flat_bvh::FlatNode<f32, 3>>                │               │               │               │         │
      ├─ 100                                79.82 µs      │ 198.9 µs      │ 123.4 µs      │ 127.9 µs      │ 100     │ 100
      ├─ 1000                               446.3 µs      │ 845.2 µs      │ 478.2 µs      │ 523.6 µs      │ 100     │ 100
      ╰─ 10000                              4.995 ms      │ 6.602 ms      │ 5.457 ms      │ 5.512 ms      │ 100     │ 100
```

As you can see, building a BVH takes a long time. Building a BVH is only useful
if the number of intersections performed on the scene exceeds the build duration.
This is the case in applications such as ray tracing and path tracing.

### Traverse

The `*_list` entries intersect a scene by traversing the raw list of triangles. This is
the most naive approach and defines the baseline. The `*_list_aabb` entries precede the
triangle test with AABB checks, which are cheap compared to the triangle-intersection
algorithms and so they filter negative results a lot faster.

The `nearest_to*` benchmarks find the closest shape to a point.

```text
traverse                                    fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ intersect_120k_list                      204.9 µs      │ 397.4 µs      │ 309 µs        │ 307 µs        │ 100     │ 100
├─ intersect_120k_list_aabb                 171.5 µs      │ 206.3 µs      │ 174.8 µs      │ 175.5 µs      │ 100     │ 100
├─ intersect_sponza_bvh                     139.8 ns      │ 2.789 µs      │ 844.8 ns      │ 931.6 ns      │ 100     │ 100
├─ intersect_sponza_iter                    92.91 µs      │ 129 µs        │ 106.4 µs      │ 107.8 µs      │ 100     │ 100
├─ intersect_sponza_list                    251.7 µs      │ 331 µs        │ 305 µs        │ 301.2 µs      │ 100     │ 100
├─ intersect_sponza_list_aabb               94.59 µs      │ 104.5 µs      │ 96.48 µs      │ 96.95 µs      │ 100     │ 100
├─ intersect_sponza_vec                     94.74 µs      │ 133.5 µs      │ 107.7 µs      │ 108.6 µs      │ 100     │ 100
├─ nearest_to_120k_list                     493.1 µs      │ 612.9 µs      │ 545.9 µs      │ 547.2 µs      │ 100     │ 100
├─ nearest_to_120k_list_aabb                170.4 µs      │ 215.7 µs      │ 176.6 µs      │ 177.3 µs      │ 100     │ 100
├─ nearest_to_sponza_bvh                    309.8 ns      │ 9.609 µs      │ 1.549 µs      │ 2.037 µs      │ 100     │ 100
├─ nearest_to_sponza_list                   479.2 µs      │ 591.6 µs      │ 516.6 µs      │ 518.8 µs      │ 100     │ 100
├─ nearest_to_sponza_list_aabb              94.39 µs      │ 110 µs        │ 99.25 µs      │ 99.32 µs      │ 100     │ 100
├─ intersect_triangles                                    │               │               │               │         │
│  ├─ Bvh<f32, 3>                                         │               │               │               │         │
│  │  ├─ 100                                74.81 ns      │ 267.3 ns      │ 111 ns        │ 122.4 ns      │ 100     │ 400
│  │  ├─ 1000                               197.3 ns      │ 514.8 ns      │ 346 ns        │ 345.1 ns      │ 100     │ 400
│  │  ╰─ 10000                              229.8 ns      │ 2.269 µs      │ 734.8 ns      │ 827.2 ns      │ 100     │ 100
│  ╰─ Vec<bvh::flat_bvh::FlatNode<f32, 3>>                │               │               │               │         │
│     ├─ 100                                72.31 ns      │ 292.3 ns      │ 109.8 ns      │ 123.4 ns      │ 100     │ 400
│     ├─ 1000                               242.3 ns      │ 1.344 µs      │ 363.5 ns      │ 378.2 ns      │ 100     │ 400
│     ╰─ 10000                              339.8 ns      │ 6.069 µs      │ 1.059 µs      │ 1.454 µs      │ 100     │ 100
╰─ nearest_to_triangles                                   │               │               │               │         │
   ├─ Bvh<f32, 3>                                         │               │               │               │         │
   │  ├─ 100                                169.8 ns      │ 382.3 ns      │ 249.8 ns      │ 253.7 ns      │ 100     │ 400
   │  ├─ 1000                               272.3 ns      │ 712.3 ns      │ 388.5 ns      │ 416.2 ns      │ 100     │ 400
   │  ╰─ 10000                              309.8 ns      │ 1.489 µs      │ 449.8 ns      │ 544.3 ns      │ 100     │ 100
   ╰─ Vec<bvh::flat_bvh::FlatNode<f32, 3>>                │               │               │               │         │
      ├─ 100                                79.81 ns      │ 1.789 µs      │ 734.8 ns      │ 737.7 ns      │ 100     │ 100
      ├─ 1000                               139.8 ns      │ 9.559 µs      │ 2.559 µs      │ 2.545 µs      │ 100     │ 100
      ╰─ 10000                              509.8 ns      │ 50.9 µs       │ 13.54 µs      │ 14.44 µs      │ 100     │ 100
```

These performance measurements show that traversing a BVH is much faster than traversing a list.

### Ray

The single ray–AABB intersection test that traversal performs at every node, measured in
isolation across 2/3/4 dimensions and `f32`/`f64`. Traversal runs this test many times
per query.

```text
ray                     fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ intersects_aabb      1.549 µs      │ 1.649 µs      │ 1.584 µs      │ 1.586 µs      │ 100     │ 100
╰─ intersects_aabb_f64                │               │               │               │         │
   ├─ 2                 1.209 µs      │ 1.309 µs      │ 1.219 µs      │ 1.224 µs      │ 100     │ 100
   ├─ 3                 2.089 µs      │ 2.279 µs      │ 2.099 µs      │ 2.107 µs      │ 100     │ 100
   ╰─ 4                 1.779 µs      │ 9.139 µs      │ 1.839 µs      │ 1.911 µs      │ 100     │ 100
```

### Optimize

These benchmarks show how long it takes to update the scene after moving a percentage of
the shapes (`10p` means `10%`).
This is where you have to differentiate between rebuilding the tree from
scratch or trying to optimize the old one (see [Optimization](#optimization)
above).
It is important to note that the randomization process here moves triangles around indiscriminately.
This will also lead to cases where the BVH would have to be restructured completely.

The `*_after_update_shapes` entries traverse a BVH kept up to date with calls to
`update_shapes` while `*_with_rebuild` uses the same triangle data but constructs a BVH
from scratch. This set of tests shows the impact of randomly moving triangles around and
producing degenerated trees.

```text
optimize                                 fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ intersect_120k_after_update_shapes                  │               │               │               │         │
│  ├─ 0                                  229.8 ns      │ 4.739 µs      │ 599.8 ns      │ 749.1 ns      │ 100     │ 100
│  ├─ 0.01                               299.8 ns      │ 1.949 µs      │ 759.8 ns      │ 771.4 ns      │ 100     │ 100
│  ├─ 0.1                                269.8 ns      │ 5.239 µs      │ 1.369 µs      │ 1.574 µs      │ 100     │ 100
│  ╰─ 0.5                                189.8 ns      │ 5.619 µs      │ 1.564 µs      │ 1.67 µs       │ 100     │ 100
├─ intersect_120k_with_rebuild                         │               │               │               │         │
│  ├─ 0                                  219.8 ns      │ 2.109 µs      │ 754.8 ns      │ 872.6 ns      │ 100     │ 100
│  ├─ 0.01                               279.8 ns      │ 4.879 µs      │ 1.559 µs      │ 1.751 µs      │ 100     │ 100
│  ├─ 0.1                                259.8 ns      │ 10.85 µs      │ 2.859 µs      │ 3.517 µs      │ 100     │ 100
│  ╰─ 0.5                                169.8 ns      │ 11.14 µs      │ 2.649 µs      │ 3.48 µs       │ 100     │ 100
├─ intersect_sponza_after_update_shapes                │               │               │               │         │
│  ├─ 0                                  129.8 ns      │ 4.989 µs      │ 924.8 ns      │ 1.2 µs        │ 100     │ 100
│  ├─ 0.01                               209.8 ns      │ 4.749 µs      │ 1.484 µs      │ 1.613 µs      │ 100     │ 100
│  ├─ 0.1                                119.8 ns      │ 26.09 µs      │ 4.849 µs      │ 6.345 µs      │ 100     │ 100
│  ╰─ 0.5                                69.81 ns      │ 20.73 µs      │ 3.714 µs      │ 4.456 µs      │ 100     │ 100
├─ intersect_sponza_with_rebuild                       │               │               │               │         │
│  ├─ 0                                  129.8 ns      │ 2.389 µs      │ 764.8 ns      │ 826.5 ns      │ 100     │ 100
│  ├─ 0.01                               189.8 ns      │ 3.039 µs      │ 804.8 ns      │ 943 ns        │ 100     │ 100
│  ├─ 0.1                                129.8 ns      │ 3.969 µs      │ 939.8 ns      │ 1.12 µs       │ 100     │ 100
│  ╰─ 0.5                                59.81 ns      │ 5.289 µs      │ 1.254 µs      │ 1.477 µs      │ 100     │ 100
├─ randomize_120k_50p                    1.317 ms      │ 1.442 ms      │ 1.326 ms      │ 1.33 ms       │ 100     │ 100
╰─ update_shapes_bvh_120k                              │               │               │               │         │
   ├─ 0                                  235.5 µs      │ 262.5 µs      │ 238.3 µs      │ 239.2 µs      │ 100     │ 100
   ├─ 0.01                               677 µs        │ 810 µs        │ 734.5 µs      │ 728.8 µs      │ 100     │ 100
   ├─ 0.1                                4.349 ms      │ 6.84 ms       │ 5.61 ms       │ 5.639 ms      │ 100     │ 100
   ╰─ 0.5                                24.8 ms       │ 34.03 ms      │ 28.79 ms      │ 28.87 ms      │ 35      │ 35
```

### Running the benchmark suite

The benchmarks use [`divan`](https://crates.io/crates/divan) and run on stable Rust:

```sh
cargo bench                    # default features (includes SIMD)
cargo bench --no-default-features --features std,rayon   # without SIMD
```

## Testing

### Unit tests

This project aspires to be fully tested, and that starts with a unit test suite. Use
`cargo test` to run all the tests. The tests automatically run in CI, too. Contributors
are expected to add tests for all new functionality.

### Proptest

This project uses [`proptest`](https://altsysrq.github.io/proptest-book/) as a second
line of defense against bugs, allowing random instances of certain tests to be tested.
These tests run alongside unit-tests.

### Fuzzer

This project uses [`cargo fuzz`](https://rust-fuzz.github.io/book/cargo-fuzz.html) as
a third line of defense against bugs, meaning that the `fuzz/` directory was generated
using `cargo fuzz init`. At the moment, there is a single fuzz target, and running
`cargo fuzz run fuzz` will fuzz until an assertion is violated. The fuzzer automatically
runs in CI, too.

## Contributing

We recommend you install the [`just`](https://github.com/casey/just) command runner, to
run [commands](/justfile) including:
- `just lint` - checks style, syntax, etc.
- `just bench` - runs the benchmarks
- `just test` - runs the unit tests and prop tests
- `just fuzz` - runs the fuzzer
