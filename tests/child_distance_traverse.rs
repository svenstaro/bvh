//! Tests for `nearest_child_traverse_iterator` and `farthest_child_traverse_iterator`.
use std::collections::HashSet;

use bvh::aabb::Bounded;
use bvh::bvh::Bvh;
use bvh::ray::Ray;
use bvh_testutil::{TBvh3, TPoint3, TVector3, UnitBox, build_empty_bh, build_some_bh};

/// Sorted ids an iterator over shapes yields
fn sorted_ids<'a>(shapes: impl Iterator<Item = &'a UnitBox>) -> Vec<i32> {
    let mut ids: Vec<i32> = shapes.map(|shape| shape.id).collect();
    ids.sort_unstable();
    ids
}

/// Verify that all three traversal APIs return exactly `expected_shapes`, ignoring order.
fn child_traverse_and_verify_set(
    ray: &Ray<f32, 3>,
    all_shapes: &[UnitBox],
    bvh: &TBvh3,
    expected_shapes: &HashSet<i32>,
) {
    let expected: Vec<i32> = sorted_ids(
        all_shapes
            .iter()
            .filter(|shape| expected_shapes.contains(&shape.id)),
    );
    assert_eq!(
        expected_shapes.len(),
        expected.len(),
        "expected shapes are not part of the scene"
    );

    // The same agreement is checked on random input by the fuzz target.
    for found in [
        sorted_ids(bvh.nearest_child_traverse_iterator(ray, all_shapes)),
        sorted_ids(bvh.farthest_child_traverse_iterator(ray, all_shapes)),
        sorted_ids(bvh.traverse(ray, all_shapes).into_iter()),
    ] {
        assert_eq!(expected, found);
    }
}

/// Verify that the child iterators return `expected_shapes` in strict distance order, which
/// they only promise while the children of the visited nodes stay disjoint. The nearest
/// iterator orders by the entry distance of a shape, the farthest one by its exit distance.
fn child_traverse_and_verify_order(
    ray: &Ray<f32, 3>,
    all_shapes: &[UnitBox],
    bvh: &TBvh3,
    expected_shapes: &HashSet<i32>,
) {
    child_traverse_and_verify_set(ray, all_shapes, bvh, expected_shapes);

    let near_it = bvh.nearest_child_traverse_iterator(ray, all_shapes);
    let far_it = bvh.farthest_child_traverse_iterator(ray, all_shapes);

    let mut prev_near_dist = -1.0;
    let mut prev_far_dist = f32::INFINITY;

    for (near_shape, far_shape) in near_it.zip(far_it) {
        let (near_dist, _) = ray.intersection_slice_for_aabb(&near_shape.aabb()).unwrap();
        let (_, far_dist) = ray.intersection_slice_for_aabb(&far_shape.aabb()).unwrap();

        assert!(prev_near_dist <= near_dist);
        assert!(prev_far_dist >= far_dist);

        prev_near_dist = near_dist;
        prev_far_dist = far_dist;
    }
}

#[test]
/// Runs some fixed child distance traversal tests on disjoint shapes.
fn test_traverse_bvh() {
    let (all_shapes, bvh) = build_some_bh::<TBvh3>();

    {
        // Define a ray which traverses the x-axis from afar.
        let ray = Ray::new(
            TPoint3::new(-1000.0, 0.0, 0.0),
            TVector3::new(1.0, 0.0, 0.0),
        );

        // It should hit everything.
        let expected_shapes = (-10..11).collect();
        child_traverse_and_verify_order(&ray, &all_shapes, &bvh, &expected_shapes);
    }

    {
        // Define a ray which intersects the x-axis diagonally.
        let ray = Ray::new(TPoint3::new(6.0, 0.5, 0.0), TVector3::new(-2.0, -1.0, 0.0));

        // It should hit exactly three boxes.
        let expected_shapes = [4, 5, 6].into_iter().collect();
        child_traverse_and_verify_order(&ray, &all_shapes, &bvh, &expected_shapes);
    }
}

/// A 3x3x3 grid of unit boxes spaced closer than their own size, so that the children of the
/// interior nodes overlap. That is the case the child ordering gives up on.
fn build_overlapping_boxes() -> (Vec<UnitBox>, TBvh3) {
    let mut shapes = Vec::new();
    let mut id = 0;
    for x in 0..3 {
        for y in 0..3 {
            for z in 0..3 {
                shapes.push(UnitBox::new(
                    id,
                    TPoint3::new(x as f32 * 0.4, y as f32 * 0.4, z as f32 * 0.4),
                ));
                id += 1;
            }
        }
    }

    let bvh = Bvh::build(&mut shapes);
    (shapes, bvh)
}

#[test]
/// Overlapping children must not cost shapes, only order. Should the iterators ever become
/// sorted here, they are stricter than documented and the docs and complexity claims of
/// `nearest_child_traverse_iterator` and `farthest_child_traverse_iterator` need updating.
fn test_overlapping_children_are_unordered_but_complete() {
    let (all_shapes, bvh) = build_overlapping_boxes();
    let ray = Ray::new(TPoint3::new(-2.0, 0.4, 0.4), TVector3::new(1.0, 0.0, 0.0));

    // Every box is hit, no matter the order.
    let expected_shapes = all_shapes.iter().map(|shape| shape.id).collect();
    child_traverse_and_verify_set(&ray, &all_shapes, &bvh, &expected_shapes);

    let near_it = bvh.nearest_child_traverse_iterator(&ray, &all_shapes);
    let far_it = bvh.farthest_child_traverse_iterator(&ray, &all_shapes);

    let near_dists: Vec<f32> = near_it
        .map(|shape| ray.intersection_slice_for_aabb(&shape.aabb()).unwrap().0)
        .collect();
    let far_dists: Vec<f32> = far_it
        .map(|shape| ray.intersection_slice_for_aabb(&shape.aabb()).unwrap().1)
        .collect();

    assert!(
        near_dists.windows(2).any(|pair| pair[0] > pair[1]),
        "nearest child traversal came back sorted: {near_dists:?}"
    );
    assert!(
        far_dists.windows(2).any(|pair| pair[0] < pair[1]),
        "farthest child traversal came back sorted: {far_dists:?}"
    );
}

#[test]
fn test_traverse_empty_bvh() {
    let (shapes, bvh) = build_empty_bh::<TBvh3>();

    // Define an arbitrary ray.
    let ray = Ray::new(TPoint3::new(0.0, 0.0, 0.0), TVector3::new(1.0, 0.0, 0.0));

    // Ensure child distance traversal doesn't panic.
    assert_eq!(
        bvh.nearest_child_traverse_iterator(&ray, &shapes).count(),
        0
    );
    assert_eq!(
        bvh.farthest_child_traverse_iterator(&ray, &shapes).count(),
        0
    );
}

#[test]
/// A single shape has no interior node to order, so it is the simplest complete traversal.
fn test_traverse_single_shape_bvh() {
    let mut shapes = vec![UnitBox::new(0, TPoint3::new(3.0, 0.0, 0.0))];
    let bvh = Bvh::build(&mut shapes);

    let ray = Ray::new(TPoint3::new(0.0, 0.0, 0.0), TVector3::new(1.0, 0.0, 0.0));
    child_traverse_and_verify_set(&ray, &shapes, &bvh, &[0].into_iter().collect());
}
