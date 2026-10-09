//! Tests extracted from `aabb_impl.rs`.
use bvh::aabb::Bounded;
use bvh_testutil::{
    TAabb3, TPoint3, TVector3, TupleVec, tuple_to_point, tuple_to_vector, tuplevec_large_strategy,
};

use float_eq::assert_float_eq;
use proptest::prelude::*;

#[test]
fn test_overflowing_aabb_center() {
    // Define two points which will be the corners of the overflowing `Aabb`
    let p1 = tuple_to_point(&(-3.288583e38, 0.0, 0.0));
    let p2 = tuple_to_point(&(5.4196525e37, 0.0, 0.0));

    // Span the `Aabb`
    let aabb = TAabb3::empty().grow(&p1).join_bounded(&p2);

    // Make sure the size actually overflows.
    assert!(aabb.size()[0].is_infinite());

    // Make sure the center does not overflow.
    assert!(aabb.center()[0].is_finite());

    // Its center should inside the `Aabb`
    assert!(aabb.contains(&aabb.center()));
}

proptest! {
    // Test properties of `Aabb` intersection.
    #[test]
    fn test_intersecting_aabbs(a: TupleVec, b: TupleVec, c: TupleVec, d: TupleVec, p: TupleVec) {
        let a = tuple_to_point(&a);
        let b = tuple_to_point(&b);
        let c = tuple_to_point(&c);
        let d = tuple_to_point(&d);
        let aabb1 = TAabb3::empty().grow(&a).join_bounded(&b);
        let aabb2 = TAabb3::empty().grow(&c).join_bounded(&d);
        if aabb1.intersects_aabb(&aabb2) {
            // For intersecting Aabb's, at least one point is shared.
            let mut closest = aabb1.center();
            for i in 0..3 {
                closest[i] = closest[i].clamp(aabb2.min[i], aabb2.max[i]);
            }
            assert!(aabb1.contains(&closest), "closest={closest:?}");
            assert!(aabb2.contains(&closest), "closest={closest:?}");
        } else {
            // For non-intersecting Aabb's, no point can't be in both Aabb's.
            let p = tuple_to_point(&p);
            for point in [a, b, c, d, p] {
                assert!(!aabb1.contains(&point) || !aabb2.contains(&point));
            }
        }
    }

    // Test whether an empty `Aabb` does not contains anything.
    #[test]
    fn test_empty_contains_nothing(tpl: TupleVec) {
        // Define a random Point
        let p = tuple_to_point(&tpl);

        // Create an empty Aabb
        let aabb = TAabb3::empty();

        // It should not contain anything
        assert!(!aabb.contains(&p));
    }

    // Test whether a default `Aabb` is empty.
    #[test]
    fn test_default_is_empty(tpl: TupleVec) {
        // Define a random Point
        let p = tuple_to_point(&tpl);

        // Create a default Aabb
        let aabb: TAabb3 = Default::default();

        // It should not contain anything
        assert!(!aabb.contains(&p));
    }

    // Test whether an `Aabb` always contains its center.
    #[test]
    fn test_aabb_contains_center(a: TupleVec, b: TupleVec) {
        // Define two points which will be the corners of the `Aabb`
        let p1 = tuple_to_point(&a);
        let p2 = tuple_to_point(&b);

        // Span the `Aabb`
        let aabb = TAabb3::empty().grow(&p1).join_bounded(&p2);

        // Its center should be inside the `Aabb`
        assert!(aabb.contains(&aabb.center()));
    }

    // Test whether the joint of two point-sets contains all the points.
    #[test]
    fn test_join_two_aabbs(a: (TupleVec, TupleVec, TupleVec, TupleVec, TupleVec),
                           b: (TupleVec, TupleVec, TupleVec, TupleVec, TupleVec))
                           {
        // Define an array of ten points
        let points = [a.0, a.1, a.2, a.3, a.4, b.0, b.1, b.2, b.3, b.4];

        // Convert these points to `Point3`
        let points = points.iter().map(tuple_to_point).collect::<Vec<TPoint3>>();

        // Create two `Aabb`s. One spanned the first five points,
        // the other by the last five points
        let aabb1 = points.iter().take(5).fold(TAabb3::empty(), |aabb, point| aabb.grow(point));
        let aabb2 = points.iter().skip(5).fold(TAabb3::empty(), |aabb, point| aabb.grow(point));

        // The `Aabb`s should contain the points by which they are spanned
        let aabb1_contains_init_five = points.iter()
            .take(5)
            .all(|point| aabb1.contains(point));
        let aabb2_contains_last_five = points.iter()
            .skip(5)
            .all(|point| aabb2.contains(point));

        // Build the joint of the two `Aabb`s
        let aabbu = aabb1.join(&aabb2);

        // The joint should contain all points
        let aabbu_contains_all = points.iter()
            .all(|point| aabbu.contains(point));

        // Return the three properties
        assert!(aabb1_contains_init_five && aabb2_contains_last_five && aabbu_contains_all);
    }

    // Test whether some points relative to the center of an `Aabb` are classified correctly.
    // Currently doesn't test `approx_contains_eps` or `contains` very well due to scaling by 0.9 and 1.1.
    #[test]
    fn test_points_relative_to_center_and_size(a in tuplevec_large_strategy(), b in tuplevec_large_strategy()) {
        // Generate some nonempty Aabb
        let aabb = TAabb3::empty()
            .grow(&tuple_to_point(&a))
            .grow(&tuple_to_point(&b));

        // Get its size and center
        let size = aabb.size();
        let size_half = size / 2.0;
        let center = aabb.center();

        // Compute the min and the max corners of the `Aabb` by hand
        let inside_ppp = center + size_half * 0.9;
        let inside_mmm = center - size_half * 0.9;

        // Generate two points which are outside the `Aabb`
        let outside_ppp = inside_ppp + size_half * 1.1;
        let outside_mmm = inside_mmm - size_half * 1.1;

        assert!(aabb.approx_contains_eps(&inside_ppp, f32::EPSILON));
        assert!(aabb.approx_contains_eps(&inside_mmm, f32::EPSILON));
        assert!(!aabb.contains(&outside_ppp));
        assert!(!aabb.contains(&outside_mmm));
    }

    // Test whether the surface of a nonempty `Aabb is always positive.
    #[test]
    fn test_surface_always_positive(a: TupleVec, b: TupleVec) {
        let aabb = TAabb3::empty()
            .grow(&tuple_to_point(&a))
            .grow(&tuple_to_point(&b));
        assert!(aabb.surface_area() >= 0.0);
    }

    // Compute and compare the surface area of an `Aabb` by hand.
    #[test]
    fn test_surface_area_cube(pos: TupleVec, size in f32::EPSILON..10e30_f32) {
        // Generate some non-empty Aabb
        let pos = tuple_to_point(&pos);
        let size_vec = TVector3::new(size, size, size);
        let aabb = TAabb3::with_bounds(pos, pos + size_vec);

        // Check its surface area
        let area_a = aabb.surface_area();
        let area_b = 6.0 * size * size;
        assert_float_eq!(area_a, area_b, rmax <= f32::EPSILON);
    }

    // Test whether the volume of a nonempty `Aabb` is always positive.
    #[test]
    fn test_volume_always_positive(a in tuplevec_large_strategy(), b in tuplevec_large_strategy()) {
        let aabb = TAabb3::empty()
            .grow(&tuple_to_point(&a))
            .grow(&tuple_to_point(&b));
        assert!(aabb.volume() >= 0.0);
    }

    // Compute and compare the volume of an `Aabb` by hand.
    #[test]
    fn test_volume_by_hand(pos in tuplevec_large_strategy(), size in tuplevec_large_strategy()) {
        // Generate some non-empty Aabb
        let pos = tuple_to_point(&pos);
        let size = tuple_to_vector(&size);
        let aabb = pos.aabb().grow(&(pos + size));

        // Check its volume
        let volume_a = aabb.volume();
        let volume_b = (size.x * size.y * size.z).abs();
        assert_float_eq!(volume_a, volume_b, rmax <= f32::EPSILON);
    }

    // Test whether generating an `Aabb` from the min and max bounds yields the same `Aabb`.
    #[test]
    fn test_create_aabb_from_indexable(a: TupleVec, b: TupleVec, p: TupleVec) {
        // Create a random point
        let point = tuple_to_point(&p);

        // Create a random `Aabb`
        let aabb = TAabb3::empty()
            .grow(&tuple_to_point(&a))
            .grow(&tuple_to_point(&b));

        // Create an `Aabb` by using the index-access method
        let aabb_by_index = TAabb3::with_bounds(aabb[0], aabb[1]);

        // The `Aabb`s should be the same
        assert!(aabb.contains(&point) == aabb_by_index.contains(&point));
    }
}
