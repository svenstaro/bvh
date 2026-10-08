//! Tests extracted from `flat_bvh.rs`.
use bvh_testutil::{
    TBvh3, TFlatBvh3, build_empty_bh, build_some_bh, nearest_to_some_bh, traverse_some_bh,
};

#[test]
/// Tests whether the building procedure succeeds in not failing.
fn test_build_flat_bvh() {
    build_some_bh::<TFlatBvh3>();
}

#[test]
/// Runs some primitive tests for intersections of a ray with a fixed scene given
/// as a `FlatBvh`.
fn test_traverse_flat_bvh() {
    traverse_some_bh::<TFlatBvh3>();
}

#[test]
/// Runs some primitive tests for distance query of a point with a fixed scene given as a [`Bvh`].
fn test_nearest_to_flat_bvh() {
    nearest_to_some_bh::<TFlatBvh3>();
}
#[test]
fn test_flatten_empty_bvh() {
    let (_, bvh) = build_empty_bh::<TBvh3>();
    let flat = bvh.flatten();
    assert!(flat.is_empty());
}
