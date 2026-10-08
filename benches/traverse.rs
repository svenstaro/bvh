//! Traversal benchmarks: ray/point queries against hierarchies and the
//! non-accelerated linear baselines they're compared against.

use bvh::aabb::Bounded;
use bvh::bounding_hierarchy::BoundingHierarchy;
use bvh::point_query::PointDistance;
use bvh_testutil::{
    TAabb3, TBvh3, TFlatBvh3, Triangle, create_n_cubes, create_ray, default_bounds,
    load_sponza_scene, next_point3,
};

fn main() {
    divan::main();
}

/// Random-ray traversal of an already-built hierarchy.
fn intersect_scene<T: BoundingHierarchy<f32, 3>>(
    bh: &T,
    bencher: divan::Bencher,
    triangles: &[Triangle],
    bounds: &TAabb3,
) {
    let mut seed = 0;
    bencher.bench_local(|| {
        let ray = create_ray(&mut seed, bounds);
        for triangle in bh.traverse(&ray, triangles) {
            divan::black_box(ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c));
        }
    });
}

/// Random nearest-shape query against an already-built hierarchy.
fn nearest_to_scene<T: BoundingHierarchy<f32, 3>>(
    bh: &T,
    bencher: divan::Bencher,
    triangles: &[Triangle],
    bounds: &TAabb3,
) {
    let mut seed = 0;
    bencher.bench_local(|| {
        let point = next_point3(&mut seed, bounds);
        divan::black_box(bh.nearest_to(divan::black_box(point), divan::black_box(triangles)));
    });
}

/// Linear scan with random rays (no acceleration structure).
fn intersect_list(bencher: divan::Bencher, triangles: &[Triangle], bounds: &TAabb3) {
    let mut seed = 0;
    bencher.bench_local(|| {
        let ray = create_ray(&mut seed, bounds);
        for triangle in triangles {
            divan::black_box(ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c));
        }
    });
}

/// As [`intersect_list`], but pre-filtering each triangle with its [`Aabb`].
fn intersect_list_aabb(bencher: divan::Bencher, triangles: &[Triangle], bounds: &TAabb3) {
    let mut seed = 0;
    bencher.bench_local(|| {
        let ray = create_ray(&mut seed, bounds);
        for triangle in triangles {
            if ray.intersects_aabb(&triangle.aabb()) {
                divan::black_box(ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c));
            }
        }
    });
}

/// Linear nearest-to scan (no acceleration structure).
fn nearest_to_list(bencher: divan::Bencher, triangles: &[Triangle], bounds: &TAabb3) {
    let mut seed = 0;
    bencher.bench_local(|| {
        let point = next_point3(&mut seed, bounds);
        let mut min_dist = f32::MAX;
        for triangle in triangles {
            let dist = divan::black_box(triangle.distance_squared(point));
            if dist < min_dist {
                min_dist = dist;
            }
        }
        divan::black_box(min_dist)
    });
}

/// As [`nearest_to_list`], but pre-filtering via point-[`Aabb`] distance.
fn nearest_to_list_aabb(bencher: divan::Bencher, triangles: &[Triangle], bounds: &TAabb3) {
    let mut seed = 0;
    bencher.bench_local(|| {
        let point = next_point3(&mut seed, bounds);
        let mut min_dist = f32::MAX;
        for triangle in triangles {
            let aabb_min_dist = divan::black_box(triangle.aabb().min_distance_squared(point));
            if aabb_min_dist < min_dist {
                let dist = divan::black_box(triangle.distance_squared(point));
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }
        divan::black_box(min_dist)
    });
}

// --- hierarchy traversal over random cube scenes ---

#[divan::bench(types = [TBvh3, TFlatBvh3], args = [100, 1_000, 10_000])]
fn intersect_triangles<T: BoundingHierarchy<f32, 3>>(bencher: divan::Bencher, n: usize) {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(n, &bounds);
    let bh = T::build(&mut triangles);
    intersect_scene(&bh, bencher, &triangles, &bounds);
}

#[divan::bench(types = [TBvh3, TFlatBvh3], args = [100, 1_000, 10_000])]
fn nearest_to_triangles<T: BoundingHierarchy<f32, 3>>(bencher: divan::Bencher, n: usize) {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(n, &bounds);
    let bh = T::build(&mut triangles);
    nearest_to_scene(&bh, bencher, &triangles, &bounds);
}

// --- hierarchy traversal over the Sponza scene ---

#[divan::bench]
fn intersect_sponza_bvh(bencher: divan::Bencher) {
    let (mut triangles, bounds) = load_sponza_scene();
    let bvh = TBvh3::build(&mut triangles);
    intersect_scene(&bvh, bencher, &triangles, &bounds);
}

#[divan::bench]
fn nearest_to_sponza_bvh(bencher: divan::Bencher) {
    let (mut triangles, bounds) = load_sponza_scene();
    let bvh = TBvh3::build(&mut triangles);
    nearest_to_scene(&bvh, bencher, &triangles, &bounds);
}

/// 128 rays per iteration, collecting hits into a [`Vec`] ([`Bvh::traverse`]).
#[divan::bench]
fn intersect_sponza_vec(bencher: divan::Bencher) {
    let (mut triangles, bounds) = load_sponza_scene();
    let bvh = TBvh3::build(&mut triangles);
    let mut seed = 0;
    bencher.bench_local(|| {
        for _ in 0..128 {
            let ray = create_ray(&mut seed, &bounds);
            for triangle in bvh.traverse(&ray, &triangles) {
                divan::black_box(ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c));
            }
        }
    });
}

/// 128 rays per iteration via the lazy iterator ([`Bvh::traverse_iterator`]).
#[divan::bench]
fn intersect_sponza_iter(bencher: divan::Bencher) {
    let (mut triangles, bounds) = load_sponza_scene();
    let bvh = TBvh3::build(&mut triangles);
    let mut seed = 0;
    bencher.bench_local(|| {
        for _ in 0..128 {
            let ray = create_ray(&mut seed, &bounds);
            for triangle in bvh.traverse_iterator(&ray, &triangles) {
                divan::black_box(ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c));
            }
        }
    });
}

// --- non-accelerated baselines ---

#[divan::bench]
fn intersect_120k_list(bencher: divan::Bencher) {
    let bounds = default_bounds();
    let triangles = create_n_cubes(10_000, &bounds);
    intersect_list(bencher, &triangles, &bounds);
}

#[divan::bench]
fn intersect_sponza_list(bencher: divan::Bencher) {
    let (triangles, bounds) = load_sponza_scene();
    intersect_list(bencher, &triangles, &bounds);
}

#[divan::bench]
fn intersect_120k_list_aabb(bencher: divan::Bencher) {
    let bounds = default_bounds();
    let triangles = create_n_cubes(10_000, &bounds);
    intersect_list_aabb(bencher, &triangles, &bounds);
}

#[divan::bench]
fn intersect_sponza_list_aabb(bencher: divan::Bencher) {
    let (triangles, bounds) = load_sponza_scene();
    intersect_list_aabb(bencher, &triangles, &bounds);
}

#[divan::bench]
fn nearest_to_120k_list(bencher: divan::Bencher) {
    let bounds = default_bounds();
    let triangles = create_n_cubes(10_000, &bounds);
    nearest_to_list(bencher, &triangles, &bounds);
}

#[divan::bench]
fn nearest_to_sponza_list(bencher: divan::Bencher) {
    let (triangles, bounds) = load_sponza_scene();
    nearest_to_list(bencher, &triangles, &bounds);
}

#[divan::bench]
fn nearest_to_120k_list_aabb(bencher: divan::Bencher) {
    let bounds = default_bounds();
    let triangles = create_n_cubes(10_000, &bounds);
    nearest_to_list_aabb(bencher, &triangles, &bounds);
}

#[divan::bench]
fn nearest_to_sponza_list_aabb(bencher: divan::Bencher) {
    let (triangles, bounds) = load_sponza_scene();
    nearest_to_list_aabb(bencher, &triangles, &bounds);
}
