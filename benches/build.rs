//! Benchmarks for building a [`BoundingHierarchy`] (serial and parallel).
//!
//! Covers the recursive [`Bvh`] and the flattened [`FlatBvh`], plus the
//! non-accelerated "build the whole thing from Sponza" baseline.

use bvh::bounding_hierarchy::BoundingHierarchy;
use bvh_testutil::{TBvh3, TFlatBvh3, create_n_cubes, default_bounds, load_sponza_scene};

fn main() {
    divan::main();
}

/// Construct a hierarchy over `n` random cubes (measures only `build`).
fn bench_build<T: BoundingHierarchy<f32, 3>>(bencher: divan::Bencher, n: usize) {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(n, &bounds);
    bencher.bench_local(|| T::build(&mut triangles));
}

/// Parallel counterpart of [`bench_build`].
#[cfg(feature = "rayon")]
fn bench_build_par<T: BoundingHierarchy<f32, 3>>(bencher: divan::Bencher, n: usize) {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(n, &bounds);
    bencher.bench_local(|| T::build_par(&mut triangles));
}

#[divan::bench(types = [TBvh3, TFlatBvh3], args = [100, 1_000, 10_000])]
fn build_triangles<T: BoundingHierarchy<f32, 3>>(bencher: divan::Bencher, n: usize) {
    bench_build::<T>(bencher, n);
}

#[cfg(feature = "rayon")]
#[divan::bench(types = [TBvh3, TFlatBvh3], args = [100, 1_000, 10_000])]
fn build_triangles_par<T: BoundingHierarchy<f32, 3>>(bencher: divan::Bencher, n: usize) {
    bench_build_par::<T>(bencher, n);
}

#[divan::bench]
fn build_sponza(bencher: divan::Bencher) {
    let (mut triangles, _) = load_sponza_scene();
    bencher.bench_local(|| TBvh3::build(&mut triangles));
}

#[cfg(feature = "rayon")]
#[divan::bench]
fn build_sponza_par(bencher: divan::Bencher) {
    let (mut triangles, _) = load_sponza_scene();
    bencher.bench_local(|| TBvh3::build_par(&mut triangles));
}

#[divan::bench]
fn flatten_120k(bencher: divan::Bencher) {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(10_000, &bounds);
    let bvh = TBvh3::build(&mut triangles);
    bencher.bench_local(|| bvh.flatten());
}
