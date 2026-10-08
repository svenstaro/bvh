//! Benchmarks for incremental re-optimization (`update_shapes`) versus a full
//! rebuild, over random cube scenes and Sponza.

use bvh_testutil::{
    TAabb3, TBvh3, Triangle, create_n_cubes, default_bounds, load_sponza_scene,
    randomly_transform_scene,
};

fn main() {
    divan::main();
}

/// Move 50% of the triangles each iteration (measures only the transform).
#[divan::bench]
fn randomize_120k_50p(bencher: divan::Bencher) {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(10_000, &bounds);
    let mut seed = 0;
    bencher.bench_local(|| {
        randomly_transform_scene(&mut triangles, 60_000, &bounds, None, &mut seed);
    });
}

/// Build once, then per iteration move `percent` of the triangles and re-run
/// `update_shapes` over the modified ones.
fn update_shapes_120k(bencher: divan::Bencher, percent: f32) {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(10_000, &bounds);
    let mut bvh = TBvh3::build(&mut triangles);
    let num_move = (triangles.len() as f32 * percent) as usize;
    let mut seed = 0;
    bencher.bench_local(|| {
        let updated =
            randomly_transform_scene(&mut triangles, num_move, &bounds, Some(10.0), &mut seed);
        bvh.update_shapes(&updated, &mut triangles);
    });
}

#[divan::bench(args = [0.0, 0.01, 0.1, 0.5])]
fn update_shapes_bvh_120k(bencher: divan::Bencher, percent: f32) {
    update_shapes_120k(bencher, percent);
}

/// Move `percent` of the triangles `iterations` times re-optimizing after each
/// move, then time traversal of the resulting hierarchy.
fn intersect_after_update_shapes(
    bencher: divan::Bencher,
    triangles: &mut [Triangle],
    bounds: &TAabb3,
    percent: f32,
    max_offset: Option<f32>,
    iterations: usize,
) {
    let mut bvh = TBvh3::build(triangles);
    let num_move = (triangles.len() as f32 * percent) as usize;
    let mut seed = 0;
    for _ in 0..iterations {
        let updated = randomly_transform_scene(triangles, num_move, bounds, max_offset, &mut seed);
        bvh.update_shapes(&updated, triangles);
    }
    intersect_traversed(&bvh, bencher, triangles, bounds, &mut seed);
}

/// Move `percent` of the triangles `iterations` times, then rebuild from
/// scratch before timing traversal. Counterpart of [`intersect_after_update_shapes`].
fn intersect_with_rebuild(
    bencher: divan::Bencher,
    triangles: &mut [Triangle],
    bounds: &TAabb3,
    percent: f32,
    max_offset: Option<f32>,
    iterations: usize,
) {
    let num_move = (triangles.len() as f32 * percent) as usize;
    let mut seed = 0;
    for _ in 0..iterations {
        randomly_transform_scene(triangles, num_move, bounds, max_offset, &mut seed);
    }
    let bvh = TBvh3::build(triangles);
    intersect_traversed(&bvh, bencher, triangles, bounds, &mut seed);
}

/// Time random-ray traversal of `bvh` (setup happens before this call).
fn intersect_traversed(
    bvh: &TBvh3,
    bencher: divan::Bencher,
    triangles: &[Triangle],
    bounds: &TAabb3,
    seed: &mut u64,
) {
    bencher.bench_local(|| {
        let ray = bvh_testutil::create_ray(seed, bounds);
        for triangle in bvh.traverse(&ray, triangles) {
            divan::black_box(triangle);
        }
    });
}

#[divan::bench(args = [0.0, 0.01, 0.1, 0.5])]
fn intersect_120k_after_update_shapes(bencher: divan::Bencher, percent: f32) {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(10_000, &bounds);
    intersect_after_update_shapes(bencher, &mut triangles, &bounds, percent, None, 10);
}

#[divan::bench(args = [0.0, 0.01, 0.1, 0.5])]
fn intersect_120k_with_rebuild(bencher: divan::Bencher, percent: f32) {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(10_000, &bounds);
    intersect_with_rebuild(bencher, &mut triangles, &bounds, percent, None, 10);
}

#[divan::bench(args = [0.0, 0.01, 0.1, 0.5])]
fn intersect_sponza_after_update_shapes(bencher: divan::Bencher, percent: f32) {
    let (mut triangles, bounds) = load_sponza_scene();
    intersect_after_update_shapes(bencher, &mut triangles, &bounds, percent, Some(0.1), 10);
}

#[divan::bench(args = [0.0, 0.01, 0.1, 0.5])]
fn intersect_sponza_with_rebuild(bencher: divan::Bencher, percent: f32) {
    let (mut triangles, bounds) = load_sponza_scene();
    intersect_with_rebuild(bencher, &mut triangles, &bounds, percent, Some(0.1), 10);
}
