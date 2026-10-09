//! Benchmarks for the ray/AABB intersection kernels (`Ray::intersects_aabb`)
//! across scalar `f32` and the `f64` SIMD kernels in 2/3/4 dimensions.

use bvh::aabb::Aabb;
use bvh::ray::Ray;
use bvh_testutil::{TAabb3, TRay3, TupleVec, tuple_to_point, tuple_to_vector};
use divan::black_box;
use nalgebra::{Point, SVector};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    divan::main();
}

/// Deterministic random ray.
fn random_ray(rng: &mut StdRng) -> TRay3 {
    let a = tuple_to_point(&rng.random::<TupleVec>());
    let b = tuple_to_vector(&rng.random::<TupleVec>());
    TRay3::new(a, b)
}

/// Deterministic random [`Aabb`].
fn random_aabb(rng: &mut StdRng) -> TAabb3 {
    let a = tuple_to_point(&rng.random::<TupleVec>());
    let b = tuple_to_point(&rng.random::<TupleVec>());
    TAabb3::empty().grow(&a).grow(&b)
}

/// A ray and 1000 boxes to test it against.
fn random_ray_and_boxes() -> (TRay3, Vec<TAabb3>) {
    let mut rng = StdRng::from_seed([0; 32]);
    let ray = random_ray(&mut rng);
    let boxes = (0..1000).map(|_| random_aabb(&mut rng)).collect::<Vec<_>>();
    black_box((ray, boxes))
}

/// A `f64` ray and boxes in `D` dimensions. Coordinates span both signs so
/// misses, hits and box-plane alignment all occur.
fn random_ray_and_boxes_f64<const D: usize>() -> (Ray<f64, D>, Vec<Aabb<f64, D>>) {
    let mut rng = StdRng::from_seed([0; 32]);
    let random = |rng: &mut StdRng| rng.random::<f64>() * 4.0 - 2.0;
    let point = |rng: &mut StdRng| Point::from(SVector::from_fn(|_, _| random(rng)));
    let ray = Ray::new(point(&mut rng), SVector::from_fn(|_, _| random(&mut rng)));
    let boxes = (0..1000)
        .map(|_| Aabb::empty().grow(&point(&mut rng)).grow(&point(&mut rng)))
        .collect::<Vec<_>>();
    black_box((ray, boxes))
}

/// Intersect one ray against 1000 `f32` boxes.
#[divan::bench]
fn intersects_aabb(bencher: divan::Bencher) {
    let (ray, boxes) = random_ray_and_boxes();
    bencher.bench(|| {
        for aabb in &boxes {
            black_box(ray.intersects_aabb(aabb));
        }
    });
}

/// Intersect one `f64` ray against 1000 boxes in each tested dimension.
#[divan::bench(consts = [2, 3, 4])]
fn intersects_aabb_f64<const D: usize>(bencher: divan::Bencher) {
    let (ray, boxes) = random_ray_and_boxes_f64::<D>();
    bencher.bench(|| {
        for aabb in &boxes {
            black_box(ray.intersects_aabb(aabb));
        }
    });
}
