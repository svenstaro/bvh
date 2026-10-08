//! Tests extracted from `iter.rs`.
use bvh::ray::Ray;
use bvh_testutil::{TBvh3, TPoint3, TVector3, UnitBox, generate_aligned_boxes};

use nalgebra::{OPoint, OVector};
use std::collections::HashSet;

/// Creates an empty [`Bvh`].
pub fn build_empty_bvh() -> ([UnitBox; 0], TBvh3) {
    let mut empty_array = [];
    let bvh = TBvh3::build(&mut empty_array);
    (empty_array, bvh)
}

/// Creates a [`Bvh`] for a fixed scene structure.
pub fn build_some_bvh() -> (Vec<UnitBox>, TBvh3) {
    let mut boxes = generate_aligned_boxes();
    let bvh = TBvh3::build(&mut boxes);
    (boxes, bvh)
}

/// Given a ray, a bounding hierarchy, the complete list of shapes in the scene and a list of
/// expected hits, verifies, whether the ray hits only the expected shapes.
fn traverse_and_verify_vec(
    ray_origin: TPoint3,
    ray_direction: TVector3,
    all_shapes: &[UnitBox],
    bvh: &TBvh3,
    expected_shapes: &HashSet<i32>,
) {
    let ray = Ray::new(ray_origin, ray_direction);
    let hit_shapes = bvh.traverse(&ray, all_shapes);

    assert_eq!(expected_shapes.len(), hit_shapes.len());
    for shape in hit_shapes {
        assert!(expected_shapes.contains(&shape.id));
    }
}

fn traverse_and_verify_iterator(
    ray_origin: TPoint3,
    ray_direction: TVector3,
    all_shapes: &[UnitBox],
    bvh: &TBvh3,
    expected_shapes: &HashSet<i32>,
) {
    let ray = Ray::new(ray_origin, ray_direction);
    let it = bvh.traverse_iterator(&ray, all_shapes);

    let mut count = 0;
    for shape in it {
        assert!(expected_shapes.contains(&shape.id));
        count += 1;
    }
    assert_eq!(expected_shapes.len(), count);
}

fn traverse_and_verify_base(
    ray_origin: TPoint3,
    ray_direction: TVector3,
    all_shapes: &[UnitBox],
    bvh: &TBvh3,
    expected_shapes: &HashSet<i32>,
) {
    traverse_and_verify_vec(ray_origin, ray_direction, all_shapes, bvh, expected_shapes);
    traverse_and_verify_iterator(ray_origin, ray_direction, all_shapes, bvh, expected_shapes);
}

/// Perform some fixed intersection tests on an empty BH structure.
pub fn traverse_empty_bvh() {
    let (empty_array, bvh) = build_empty_bvh();
    traverse_and_verify_base(
        OPoint::origin(),
        OVector::x(),
        &empty_array,
        &bvh,
        &HashSet::new(),
    );
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
        traverse_and_verify_base(origin, direction, &all_shapes, &bvh, &expected_shapes);
    }

    {
        // Define a ray which traverses the y-axis from afar.
        let origin = TPoint3::new(0.0, -1000.0, 0.0);
        let direction = TVector3::new(0.0, 1.0, 0.0);

        // It should hit only one box.
        let mut expected_shapes = HashSet::new();
        expected_shapes.insert(0);
        traverse_and_verify_base(origin, direction, &all_shapes, &bvh, &expected_shapes);
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
        traverse_and_verify_base(origin, direction, &all_shapes, &bvh, &expected_shapes);
    }
}

#[test]
/// Runs some primitive tests for intersections of a ray with a fixed scene given as a Bvh.
fn test_traverse_bvh() {
    traverse_empty_bvh();
    traverse_some_bvh();
}
