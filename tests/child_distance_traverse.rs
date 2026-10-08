//! Tests extracted from `child_distance_traverse.rs`.
use bvh::aabb::Bounded;
use bvh::bvh::Bvh;
use bvh::ray::Ray;
use bvh_testutil::{TBvh3, TPoint3, TVector3, UnitBox, generate_aligned_boxes};

use std::collections::HashSet;

/// Create a `Bvh` for a fixed scene structure.
pub fn build_some_bvh() -> (Vec<UnitBox>, TBvh3) {
    let mut boxes = generate_aligned_boxes();
    let bvh = Bvh::build(&mut boxes);
    (boxes, bvh)
}

/// Create a `Bvh` for an empty scene structure.
pub fn build_empty_bvh() -> (Vec<UnitBox>, TBvh3) {
    let mut boxes = Vec::new();
    let bvh = Bvh::build(&mut boxes);
    (boxes, bvh)
}

fn traverse_distance_and_verify_order(
    ray_origin: TPoint3,
    ray_direction: TVector3,
    all_shapes: &[UnitBox],
    bvh: &TBvh3,
    expected_shapes: &HashSet<i32>,
) {
    let ray = Ray::new(ray_origin, ray_direction);
    let near_it = bvh.nearest_traverse_iterator(&ray, all_shapes);
    let far_it = bvh.farthest_traverse_iterator(&ray, all_shapes);

    let mut count = 0;
    let mut prev_near_dist = -1.0;
    let mut prev_far_dist = f32::INFINITY;

    for (near_shape, far_shape) in near_it.zip(far_it) {
        let (intersect_near_dist, _) = ray.intersection_slice_for_aabb(&near_shape.aabb()).unwrap();
        let (intersect_far_dist, _) = ray.intersection_slice_for_aabb(&far_shape.aabb()).unwrap();

        assert!(expected_shapes.contains(&near_shape.id));
        assert!(expected_shapes.contains(&far_shape.id));
        assert!(prev_near_dist <= intersect_near_dist);
        assert!(prev_far_dist >= intersect_far_dist);

        count += 1;
        prev_near_dist = intersect_near_dist;
        prev_far_dist = intersect_far_dist;
    }
    assert_eq!(expected_shapes.len(), count);
}

/// Perform some fixed intersection tests on BH structures.
pub fn traverse_some_bvh() {
    let (all_shapes, bvh) = build_some_bvh();

    {
        // Define a ray which traverses the x-axis from afar.
        let origin = TPoint3::new(-1000.0, 0.0, 0.0);
        let direction = TVector3::new(1.0, 0.0, 0.0);
        let mut expected_shapes = HashSet::new();

        // It should hit everything.
        for id in -10..11 {
            expected_shapes.insert(id);
        }
        traverse_distance_and_verify_order(origin, direction, &all_shapes, &bvh, &expected_shapes);
    }

    {
        // Define a ray which intersects the x-axis diagonally.
        let origin = TPoint3::new(6.0, 0.5, 0.0);
        let direction = TVector3::new(-2.0, -1.0, 0.0);

        // It should hit exactly three boxes.
        let mut expected_shapes = HashSet::new();
        expected_shapes.insert(4);
        expected_shapes.insert(5);
        expected_shapes.insert(6);
        traverse_distance_and_verify_order(origin, direction, &all_shapes, &bvh, &expected_shapes);
    }
}

#[test]
/// Runs some primitive tests for intersections of a ray with a fixed scene given as a Bvh.
fn test_traverse_bvh() {
    traverse_some_bvh();
}

#[test]
fn test_traverse_empty_bvh() {
    let (shapes, bvh) = build_empty_bvh();

    // Define an arbitrary ray.
    let origin = TPoint3::new(0.0, 0.0, 0.0);
    let direction = TVector3::new(1.0, 0.0, 0.0);
    let ray = Ray::new(origin, direction);

    // Ensure distance traversal doesn't panic.
    assert_eq!(bvh.nearest_traverse_iterator(&ray, &shapes).count(), 0);
    assert_eq!(bvh.farthest_traverse_iterator(&ray, &shapes).count(), 0);
}
