//! Tests extracted from `bvh_impl.rs`.

use bvh::bounding_hierarchy::BoundingHierarchy;

use bvh_testutil::{
    TBvh3, TBvhNode3, TPoint3, TRay3, TVector3, UnitBox, build_empty_bh, build_some_bh,
    nearest_to_some_bh, traverse_some_bh,
};

#[test]
/// Tests whether the building procedure succeeds in not failing.
fn test_build_bvh() {
    build_some_bh::<TBvh3>();
}

#[test]
fn test_empty_bvh_is_consistent() {
    let (shapes, bvh) = build_empty_bh::<TBvh3>();
    bvh.assert_consistent(&shapes);
    assert!(bvh.is_consistent(&shapes));
}

#[test]
fn test_empty_bvh_is_tight() {
    let (_, bvh) = build_empty_bh::<TBvh3>();
    bvh.assert_tight();
}

#[test]
/// Runs some primitive tests for intersections of a ray with a fixed scene given as a [`Bvh`].
fn test_traverse_bvh() {
    traverse_some_bh::<TBvh3>();
}

#[test]
/// Runs some primitive tests for distance query of a point with a fixed scene given as a [`Bvh`].
fn test_nearest_to_bvh() {
    nearest_to_some_bh::<TBvh3>();
}

#[test]
/// Verify contents of the bounding hierarchy for a fixed scene structure
fn test_bvh_shape_indices() {
    use std::collections::HashSet;

    let (all_shapes, bh) = build_some_bh::<TBvh3>();

    // It should find all shape indices.
    let expected_shapes: HashSet<_> = (0..all_shapes.len()).collect();
    let mut found_shapes = HashSet::new();

    for node in bh.nodes.iter() {
        match *node {
            TBvhNode3::Node { .. } => {
                assert_eq!(node.shape_index(), None);
            }
            TBvhNode3::Leaf { .. } => {
                found_shapes.insert(
                    node.shape_index()
                        .expect("getting a shape index from a leaf node"),
                );
            }
        }
    }

    assert_eq!(expected_shapes, found_shapes);
}

#[test]
#[cfg(feature = "rayon")]
/// Tests whether the building procedure succeeds in not failing.
fn test_build_bvh_rayon() {
    use bvh_testutil::build_some_bh_rayon;

    build_some_bh_rayon::<TBvh3>();
}

#[test]
#[cfg(feature = "rayon")]
/// Runs some primitive tests for intersections of a ray with a fixed scene given as a [`Bvh`].
fn test_traverse_bvh_rayon() {
    use bvh_testutil::traverse_some_bh_rayon;

    traverse_some_bh_rayon::<TBvh3>();
}

#[test]
#[cfg(feature = "rayon")]
/// Verify contents of the bounding hierarchy for a fixed scene structure
fn test_bvh_shape_indices_rayon() {
    use std::collections::HashSet;

    use bvh_testutil::build_some_bh_rayon;

    let (all_shapes, bh) = build_some_bh_rayon::<TBvh3>();

    // It should find all shape indices.
    let expected_shapes: HashSet<_> = (0..all_shapes.len()).collect();
    let mut found_shapes = HashSet::new();

    for node in bh.nodes.iter() {
        match *node {
            TBvhNode3::Node { .. } => {
                assert_eq!(node.shape_index(), None);
            }
            TBvhNode3::Leaf { .. } => {
                found_shapes.insert(
                    node.shape_index()
                        .expect("getting a shape index from a leaf node"),
                );
            }
        }
    }

    assert_eq!(expected_shapes, found_shapes);
}

/// A single-node BVH is special, since the root node is a leaf node. Make sure
/// the root node isn't unconditionally returned when it isn't intersected.
#[test]
fn test_traverse_one_node_bvh_no_intersection() {
    let mut boxes = vec![UnitBox::new(0, TPoint3::new(0.0, 1.0, 2.0))];
    let ray = TRay3::new(TPoint3::new(0.0, 0.0, 0.0), TVector3::new(1.0, 0.0, 0.0));
    let bvh = TBvh3::build(&mut boxes);

    assert!(bvh.traverse(&ray, &boxes).is_empty());
    assert!(bvh.traverse_iterator(&ray, &boxes).next().is_none());
    assert!(bvh.nearest_traverse_iterator(&ray, &boxes).next().is_none());
    assert!(bvh.flatten().traverse(&ray, &boxes).is_empty())
}

/// Make sure the root node can be returned when it is intersected.
#[test]
fn test_traverse_one_node_bvh_intersection() {
    let mut boxes = vec![UnitBox::new(0, TPoint3::new(10.0, 0.0, 0.0))];
    let ray = TRay3::new(TPoint3::new(0.0, 0.0, 0.0), TVector3::new(1.0, 0.0, 0.0));
    let bvh = TBvh3::build(&mut boxes);

    assert_eq!(bvh.traverse(&ray, &boxes).len(), 1);
    assert_eq!(bvh.traverse_iterator(&ray, &boxes).count(), 1);
    assert_eq!(bvh.nearest_traverse_iterator(&ray, &boxes).count(), 1);
    assert_eq!(bvh.flatten().traverse(&ray, &boxes).len(), 1)
}
