//! Tests extracted from `ray_impl.rs`.
use bvh::aabb::Bounded;

use bvh_testutil::{
    TAabb3, TPoint3, TRay3, TVector3, TupleVec, UnitBox, tuple_to_point, tuplevec_small_strategy,
};
use core::cmp;
use proptest::prelude::*;

/// Generates a random [`Ray`] which points at at a random [`Aabb`].
fn gen_ray_to_aabb(data: (TupleVec, TupleVec, TupleVec)) -> (TRay3, TAabb3) {
    // Generate a random `Aabb`
    let aabb = TAabb3::empty()
        .grow(&tuple_to_point(&data.0))
        .grow(&tuple_to_point(&data.1));

    // Get its center
    let center = aabb.center();

    // Generate random ray pointing at the center
    let pos = tuple_to_point(&data.2);
    let ray = TRay3::new(pos, center - pos);
    (ray, aabb)
}

/// The SIMD kernels selected for `f64` must agree with the scalar
/// [`Ray::intersection_slice_for_aabb`] reference in every supported
/// dimension.
#[test]
fn intersects_aabb_f64_matches_scalar_reference() {
    use bvh::{aabb::Aabb, ray::Ray};
    use nalgebra::{Point, SVector};
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    fn random(rng: &mut StdRng) -> f64 {
        rng.random::<f64>() * 4.0 - 2.0
    }

    // A random ray/box pair in `D` dimensions.
    fn check<const D: usize>(rng: &mut StdRng) {
        let point = |rng: &mut StdRng| Point::from(SVector::from_fn(|_, _| random(rng)));
        let ray: Ray<f64, D> = Ray::new(point(rng), SVector::from_fn(|_, _| random(rng)));
        let aabb: Aabb<f64, D> = Aabb::empty().grow(&point(rng)).grow(&point(rng));
        assert_eq!(
            ray.intersects_aabb(&aabb),
            ray.intersection_slice_for_aabb(&aabb).is_some(),
            "ray: {ray:?}, aabb: {aabb:?}"
        );
    }

    let mut rng = StdRng::from_seed([0; 32]);
    for _ in 0..1000 {
        check::<2>(&mut rng);
        check::<3>(&mut rng);
        check::<4>(&mut rng);
    }

    // A ray aligned with a box plane produces NaN slab values; every
    // kernel must treat this as a non-intersection.
    fn check_plane_aligned<const D: usize>() {
        let ray: Ray<f64, D> = Ray::new(
            Point::from(SVector::from_fn(|i, _| if i == 1 { -5.0 } else { 0.0 })),
            SVector::from_fn(|i, _| if i == 1 { 1.0 } else { 0.0 }),
        );
        let aabb: Aabb<f64, D> = Aabb::with_bounds(
            Point::from(SVector::from_fn(|i, _| if i == 1 { -1.0 } else { 0.0 })),
            Point::from(SVector::from_fn(|_, _| 1.0)),
        );
        assert!(!ray.intersects_aabb(&aabb));
        assert!(ray.intersection_slice_for_aabb(&aabb).is_none());
    }
    check_plane_aligned::<2>();
    check_plane_aligned::<3>();
    check_plane_aligned::<4>();
}

/// Make sure a ray can intersect an AABB with no depth.
#[test]
fn ray_hits_zero_depth_aabb() {
    let origin = TPoint3::new(0.0, 0.0, 0.0);
    let direction = TVector3::new(0.0, 0.0, 1.0);
    let ray = TRay3::new(origin, direction);
    let min = TPoint3::new(-1.0, -1.0, 1.0);
    let max = TPoint3::new(1.0, 1.0, 1.0);
    let aabb = TAabb3::with_bounds(min, max);
    assert!(ray.intersects_aabb(&aabb));
}

/// Ensure slice has correct min and max distance for a particular case.
#[test]
fn test_ray_slice_distance_accuracy() {
    let aabb = TAabb3::empty()
        .grow(&TPoint3::new(-3.0, -4.0, -5.0))
        .grow(&TPoint3::new(-6.0, -8.0, 5.0));
    let ray = TRay3::new(
        TPoint3::new(2.0, 2.0, 2.0),
        TVector3::new(-5.0, -8.66666, -3.666666),
    );
    let expected_min = 10.6562;
    let expected_max = 12.3034;
    let (min, max) = ray.intersection_slice_for_aabb(&aabb.aabb()).unwrap();
    assert!((min - expected_min).abs() < 0.01);
    assert!((max - expected_max).abs() < 0.01);
}

/// Ensure no slice is returned when the ray is parallel to AABB faces but
/// doesn't hit, which is a special case due to infinities in the computation.
#[test]
fn test_parallel_ray_slice() {
    let aabb = UnitBox::new(0, TPoint3::new(-50.0, -50.0, -25.0));
    let ray = TRay3::new(
        TPoint3::new(-50.0, -50.0, -50.0),
        TVector3::new(1.0, 0.0, 0.0),
    );
    assert!(ray.intersection_slice_for_aabb(&aabb.aabb()).is_none());
}

/// Ensure no slice is returned when the ray is in the plane of an AABB
/// face, which is a special case due to NaN's in the computation.
#[test]
fn test_in_plane_ray_slice() {
    let aabb = UnitBox::new(0, TPoint3::new(0.0, 0.0, 0.0)).aabb();
    let ray = TRay3::new(TPoint3::new(0.0, 0.0, -0.5), TVector3::new(1.0, 0.0, 0.0));
    assert!(!ray.intersects_aabb(&aabb));
    assert!(ray.intersection_slice_for_aabb(&aabb).is_none());

    // Test a different ray direction, to ensure that order of `fast_max` (relevant
    // to the result when NaN's are involved) doesn't matter.
    let ray = TRay3::new(TPoint3::new(0.0, 0.5, 0.0), TVector3::new(0.0, 0.0, 1.0));
    assert!(!ray.intersects_aabb(&aabb));
    assert!(ray.intersection_slice_for_aabb(&aabb).is_none());
}

proptest! {
    // Test whether a `Ray` which points at the center of an `Aabb` intersects it.
    #[test]
    fn test_ray_points_at_aabb_center(data in (tuplevec_small_strategy(),
                                               tuplevec_small_strategy(),
                                               tuplevec_small_strategy())) {
        let (ray, aabb) = gen_ray_to_aabb(data);
        assert!(ray.intersects_aabb(&aabb));
    }

    // Test whether a `Ray` which points away from the center of an `Aabb`
    // does not intersect it, unless its origin is inside the `Aabb`.
    #[test]
    fn test_ray_points_from_aabb_center(data in (tuplevec_small_strategy(),
                                                 tuplevec_small_strategy(),
                                                 tuplevec_small_strategy())) {
        let (mut ray, aabb) = gen_ray_to_aabb(data);

        // Invert the direction of the ray
        ray.direction = -ray.direction;
        ray.inv_direction = -ray.inv_direction;
        assert!(!ray.intersects_aabb(&aabb) || aabb.contains(&ray.origin));
    }

    // Test whether a `Ray` which points at the center of an `Aabb` takes intersection slice.
    #[test]
    fn test_ray_slice_at_aabb_center(data in (tuplevec_small_strategy(),
                                               tuplevec_small_strategy(),
                                               tuplevec_small_strategy())) {
        let (ray, aabb) = gen_ray_to_aabb(data);
        let (start_dist, end_dist) = ray.intersection_slice_for_aabb(&aabb).unwrap();
        assert!(start_dist < end_dist);
        assert!(start_dist >= 0.0);
    }

    // Test whether a `Ray` which points away from the center of an `Aabb`
    // cannot take intersection slice of it, unless its origin is inside the `Aabb`.
    #[test]
    fn test_ray_slice_from_aabb_center(data in (tuplevec_small_strategy(),
                                                 tuplevec_small_strategy(),
                                                 tuplevec_small_strategy())) {
        let (mut ray, aabb) = gen_ray_to_aabb(data);

        // Invert the direction of the ray
        ray.direction = -ray.direction;
        ray.inv_direction = -ray.inv_direction;

        let slice = ray.intersection_slice_for_aabb(&aabb);
        if aabb.contains(&ray.origin) {
            let (start_dist, end_dist) = slice.unwrap();
            // ray inside of aabb
            assert!(start_dist < end_dist);
            assert!(start_dist >= 0.0);
        } else {
            // ray outside of aabb and doesn't intersect it
            assert!(slice.is_none());
        }
    }

    // Test whether a `Ray` which points at the center of a triangle
    // intersects it, unless it sees the back face, which is culled.
    #[test]
    fn test_ray_hits_triangle(a in tuplevec_small_strategy(),
                              b in tuplevec_small_strategy(),
                              c in tuplevec_small_strategy(),
                              origin in tuplevec_small_strategy(),
                              u: u16,
                              v: u16) {
        // Define a triangle, u/v vectors and its normal
        let triangle = (tuple_to_point(&a), tuple_to_point(&b), tuple_to_point(&c));
        let u_vec = triangle.1 - triangle.0;
        let v_vec = triangle.2 - triangle.0;
        let normal = u_vec.cross(&v_vec);

        // Get some u and v coordinates such that u+v <= 1
        let u = u % 101;
        let v = cmp::min(100 - u, v % 101);
        let u = u as f32 / 100.0;
        let v = v as f32 / 100.0;

        // Define some point on the triangle
        let point_on_triangle = triangle.0 + u * u_vec + v * v_vec;

        // Define a ray which points at the triangle
        let origin = tuple_to_point(&origin);
        let ray = TRay3::new(origin, point_on_triangle - origin);
        let on_back_side = normal.dot(&(ray.origin - triangle.0)) <= 0.0;

        // Perform the intersection test
        let intersects = ray.intersects_triangle(&triangle.0, &triangle.1, &triangle.2);
        let uv_sum = intersects.u + intersects.v;

        // Either the intersection is in the back side (including the triangle-plane)
        if on_back_side {
            // Intersection must be INFINITY, u and v are undefined
            assert!(intersects.distance == f32::INFINITY);
        } else {
            // Or it is on the front side
            // Either the intersection is inside the triangle, which it should be
            // for all u, v such that u+v <= 1.0
            let intersection_inside = (0.0..=1.0).contains(&uv_sum) && intersects.distance < f32::INFINITY;

            // Or the input data was close to the border
            let close_to_border =
                u.abs() < f32::EPSILON || (u - 1.0).abs() < f32::EPSILON || v.abs() < f32::EPSILON ||
                (v - 1.0).abs() < f32::EPSILON || (u + v - 1.0).abs() < f32::EPSILON;

            #[cfg(feature = "std")]
            if !(intersection_inside || close_to_border) {
                use std::println;

                println!("uvsum {uv_sum}");
                println!("intersects.0 {}", intersects.distance);
                println!("intersects.1 (u) {}", intersects.u);
                println!("intersects.2 (v) {}", intersects.v);
                println!("u {u}");
                println!("v {v}");
            }

            assert!(intersection_inside || close_to_border);
        }
    }
}
