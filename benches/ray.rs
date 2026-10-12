//! Benchmarks for the ray/AABB intersection kernels (`Ray::intersects_aabb`)
//! across scalar `f32` and the `f64` SIMD kernels in 2/3/4 dimensions.

use bvh::aabb::Aabb;
use bvh_testutil::{random_ray_and_aabbs, seeded_rng, uniform_bounds, unit_bounds};
use divan::black_box;

/// Boxes each ray is intersected against.
const N_BOXES: usize = 1000;

fn main() {
    divan::main();
}

/// Intersect one ray against 1000 `f32` boxes.
#[divan::bench]
fn intersects_aabb(bencher: divan::Bencher) {
    let (ray, boxes) = black_box(random_ray_and_aabbs(
        &mut seeded_rng(0),
        &unit_bounds(),
        N_BOXES,
    ));
    bencher.bench(|| {
        for aabb in &boxes {
            black_box(ray.intersects_aabb(aabb));
        }
    });
}

/// Intersect one `f64` ray against 1000 boxes in each tested dimension. The
/// bounds span both signs so misses, hits and box-plane alignment all occur.
#[divan::bench(consts = [2, 3, 4])]
fn intersects_aabb_f64<const D: usize>(bencher: divan::Bencher) {
    let bounds: Aabb<f64, D> = uniform_bounds(-2.0, 2.0);
    let (ray, boxes) = black_box(random_ray_and_aabbs(&mut seeded_rng(0), &bounds, N_BOXES));
    bencher.bench(|| {
        for aabb in &boxes {
            black_box(ray.intersects_aabb(aabb));
        }
    });
}
