//! Tests extracted from `optimization.rs`.
use bvh::bounding_hierarchy::BHShape;
use bvh_testutil::{
    TBvh3, TBvhNode3, TPoint3, UnitBox, build_some_bh, create_n_cubes, default_bounds,
    randomly_transform_scene,
};
use std::collections::HashSet;

#[test]
/// Tests whether a Bvh is still consistent after a few optimization calls.
fn test_consistent_after_update_shapes() {
    let (mut shapes, mut bvh) = build_some_bh::<TBvh3>();
    shapes[0].pos = TPoint3::new(10.0, 1.0, 2.0);
    shapes[1].pos = TPoint3::new(-10.0, -10.0, 10.0);
    shapes[2].pos = TPoint3::new(-10.0, 10.0, 10.0);
    shapes[3].pos = TPoint3::new(-10.0, 10.0, -10.0);
    shapes[4].pos = TPoint3::new(11.0, 1.0, 2.0);
    shapes[5].pos = TPoint3::new(11.0, 2.0, 2.0);
    let refit_shape_indices: Vec<_> = (0..6).collect();
    bvh.update_shapes(&refit_shape_indices, &mut shapes);
    bvh.assert_consistent(&shapes);
}

#[test]
/// Test whether a simple update on a simple [`Bvh]` yields the expected optimization result.
fn test_update_shapes_simple_update() {
    let mut shapes = vec![
        UnitBox::new(0, TPoint3::new(-50.0, 0.0, 0.0)),
        UnitBox::new(1, TPoint3::new(-40.0, 0.0, 0.0)),
        UnitBox::new(2, TPoint3::new(50.0, 0.0, 0.0)),
    ];

    let mut bvh = TBvh3::build(&mut shapes);
    #[cfg(feature = "std")]
    bvh.pretty_print();

    // Assert that SAH joined shapes #0 and #1.
    {
        let left = &shapes[0];
        let moving = &shapes[1];

        match (
            &bvh.nodes[left.bh_node_index()],
            &bvh.nodes[moving.bh_node_index()],
        ) {
            (
                &TBvhNode3::Leaf {
                    parent_index: left_parent_index,
                    ..
                },
                &TBvhNode3::Leaf {
                    parent_index: moving_parent_index,
                    ..
                },
            ) => {
                assert_eq!(moving_parent_index, left_parent_index);
            }
            _ => panic!(),
        }
    }

    // Move the first shape so that it is closer to shape #2.
    shapes[1].pos = TPoint3::new(40.0, 0.0, 0.0);
    let refit_shape_indices: HashSet<usize> = (1..2).collect();
    bvh.update_shapes(&refit_shape_indices, &mut shapes);
    #[cfg(feature = "std")]
    bvh.pretty_print();
    bvh.assert_consistent(&shapes);

    // Assert that now SAH joined shapes #1 and #2.
    {
        let moving = &shapes[1];
        let right = &shapes[2];

        match (
            &bvh.nodes[right.bh_node_index()],
            &bvh.nodes[moving.bh_node_index()],
        ) {
            (
                &TBvhNode3::Leaf {
                    parent_index: right_parent_index,
                    ..
                },
                &TBvhNode3::Leaf {
                    parent_index: moving_parent_index,
                    ..
                },
            ) => {
                assert_eq!(moving_parent_index, right_parent_index);
            }
            _ => panic!(),
        }
    }
}

#[test]
/// Test optimizing [`Bvh`] after randomizing 50% of the shapes.
fn test_update_shapes_bvh_12k_75p() {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(1_000, &bounds);

    let mut bvh = TBvh3::build(&mut triangles);

    // The initial Bvh should be consistent.
    bvh.assert_consistent(&triangles);
    bvh.assert_tight();

    // After moving triangles, the Bvh should be inconsistent, because the shape `Aabb`s do not
    // match the tree entries.
    let mut seed = 0;

    let updated = randomly_transform_scene(&mut triangles, 9_000, &bounds, None, &mut seed);
    assert!(!bvh.is_consistent(&triangles), "Bvh is consistent.");

    // After fixing the `Aabb` consistency should be restored.
    bvh.update_shapes(&updated, &mut triangles);
    bvh.assert_consistent(&triangles);
    bvh.assert_tight();
}
