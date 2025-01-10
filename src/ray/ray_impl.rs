//! This module defines a Ray structure and intersection algorithms
//! for axis aligned bounding boxes and triangles.

use crate::aabb::IntersectsAabb;
use crate::utils::{fast_max, fast_min};
use crate::{aabb::Aabb, bounding_hierarchy::BHValue};
use nalgebra::{
    ClosedAddAssign, ClosedMulAssign, ClosedSubAssign, ComplexField, Point, SVector, SimdPartialOrd,
};
use num::{Float, One, Zero};

use super::intersect_default::RayIntersection;

/// A struct which defines a ray and some of its cached values.
#[derive(Debug, Clone, Copy)]
pub struct Ray<T: BHValue, const D: usize> {
    /// The ray origin.
    pub origin: Point<T, D>,

    /// The ray direction.
    pub direction: SVector<T, D>,

    /// Inverse (1/x) ray direction. Cached for use in [`Aabb`] intersections.
    ///
    /// [`Aabb`]: struct.Aabb.html
    ///
    pub inv_direction: SVector<T, D>,
}

/// A struct which is returned by the [`Ray::intersects_triangle()`] method.
pub struct Intersection<T> {
    /// Distance from the ray origin to the intersection point.
    pub distance: T,

    /// U coordinate of the intersection.
    pub u: T,

    /// V coordinate of the intersection.
    pub v: T,
}

impl<T> Intersection<T> {
    /// Constructs an [`Intersection`]. `distance` should be set to positive infinity,
    /// if the intersection does not occur.
    pub fn new(distance: T, u: T, v: T) -> Intersection<T> {
        Intersection { distance, u, v }
    }
}

impl<T: BHValue, const D: usize> Ray<T, D> {
    /// Creates a new [`Ray`] from an `origin` and a `direction`.
    /// `direction` will be normalized.
    ///
    /// # Examples
    /// ```
    /// use bvh::ray::Ray;
    /// use nalgebra::{Point3,Vector3};
    ///
    /// let origin = Point3::new(0.0,0.0,0.0);
    /// let direction = Vector3::new(1.0,0.0,0.0);
    /// let ray = Ray::new(origin, direction);
    ///
    /// assert_eq!(ray.origin, origin);
    /// assert_eq!(ray.direction, direction);
    /// ```
    ///
    /// [`Ray`]: struct.Ray.html
    ///
    pub fn new(origin: Point<T, D>, direction: SVector<T, D>) -> Ray<T, D>
    where
        T: One + ComplexField,
    {
        let direction = direction.normalize();
        Ray {
            origin,
            direction,
            inv_direction: direction.map(|x| T::one() / x),
        }
    }

    /// Tests the intersection of a [`Ray`] with an [`Aabb`] using the optimized algorithm
    /// from [this paper](http://www.cs.utah.edu/~awilliam/box/box.pdf).
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::Aabb;
    /// use bvh::ray::Ray;
    /// use nalgebra::{Point3,Vector3};
    ///
    /// let origin = Point3::new(0.0,0.0,0.0);
    /// let direction = Vector3::new(1.0,0.0,0.0);
    /// let ray = Ray::new(origin, direction);
    ///
    /// let point1 = Point3::new(99.9,-1.0,-1.0);
    /// let point2 = Point3::new(100.1,1.0,1.0);
    /// let aabb = Aabb::with_bounds(point1, point2);
    ///
    /// assert!(ray.intersects_aabb(&aabb));
    /// ```
    ///
    /// [`Ray`]: struct.Ray.html
    /// [`Aabb`]: struct.Aabb.html
    ///
    pub fn intersects_aabb(&self, aabb: &Aabb<T, D>) -> bool
    where
        T: ClosedSubAssign + ClosedMulAssign + Zero + PartialOrd + SimdPartialOrd,
    {
        self.ray_intersects_aabb(aabb)
    }

    /// Intersect [`Aabb`] by [`Ray`]
    /// Returns slice of intersections, two numbers `T`
    /// where the first number is the distance from [`Ray`] to the nearest intersection point
    /// and the second number is the distance from [`Ray`] to the farthest intersection point
    ///
    /// If there are no intersections, it returns negative one for both distances
    pub fn intersection_slice_for_aabb(&self, aabb: &Aabb<T, D>) -> (T, T)
    where
        T: BHValue,
    {
        // https://iquilezles.org/articles/intersectors/
        let mut n = self.origin.coords - aabb.center().coords;
        n.component_mul_assign(&self.inv_direction);

        let k = self.inv_direction.abs().component_mul(&aabb.half_size());
        let t1 = -n - k;
        let t2 = -n + k;

        let entry_distance = t1
            .iter()
            .skip(1)
            .fold(t1[0], |acc, x| -> T { fast_max(*x, acc) });
        let exit_distance = t2.iter().skip(1).fold(t2[0], |acc, x| fast_min(*x, acc));

        // no intersection
        if entry_distance > exit_distance || exit_distance < T::zero() {
            return (T::from_f32(-1.0).unwrap(), T::from_f32(-1.0).unwrap());
        }

        (fast_max(entry_distance, T::zero()), exit_distance)
    }

    /// Implementation of the
    /// [Möller-Trumbore triangle/ray intersection algorithm](https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm).
    /// Returns the distance to the intersection, as well as
    /// the u and v coordinates of the intersection.
    /// The distance is set to +INFINITY if the ray does not intersect the triangle, or hits
    /// it from behind.
    #[allow(clippy::many_single_char_names)]
    pub fn intersects_triangle(
        &self,
        a: &Point<T, D>,
        b: &Point<T, D>,
        c: &Point<T, D>,
    ) -> Intersection<T>
    where
        T: ClosedAddAssign + ClosedSubAssign + ClosedMulAssign + Zero + One + Float,
    {
        let a_to_b = *b - *a;
        let a_to_c = *c - *a;

        // Begin calculating determinant - also used to calculate u parameter
        // u_vec lies in view plane
        // length of a_to_c in view_plane = |u_vec| = |a_to_c|*sin(a_to_c, dir)
        let u_vec = self.direction.cross(&a_to_c);

        // If determinant is near zero, ray lies in plane of triangle
        // The determinant corresponds to the parallelepiped volume:
        // det = 0 => [dir, a_to_b, a_to_c] not linearly independant
        let det = a_to_b.dot(&u_vec);

        // Only testing positive bound, thus enabling backface culling
        // If backface culling is not desired write:
        // det < EPSILON && det > -EPSILON
        if det < T::epsilon() {
            return Intersection::new(T::infinity(), T::zero(), T::zero());
        }

        let inv_det = T::one() / det;

        // Vector from point a to ray origin
        let a_to_origin = self.origin - *a;

        // Calculate u parameter
        let u = a_to_origin.dot(&u_vec) * inv_det;

        // Test bounds: u < 0 || u > 1 => outside of triangle
        if !(T::zero()..=T::one()).contains(&u) {
            return Intersection::new(T::infinity(), u, T::zero());
        }

        // Prepare to test v parameter
        let v_vec = a_to_origin.cross(&a_to_b);

        // Calculate v parameter and test bound
        let v = self.direction.dot(&v_vec) * inv_det;
        // The intersection lies outside of the triangle
        if v < T::zero() || u + v > T::one() {
            return Intersection::new(T::infinity(), u, v);
        }

        let dist = a_to_c.dot(&v_vec) * inv_det;

        if dist > T::epsilon() {
            Intersection::new(dist, u, v)
        } else {
            Intersection::new(T::infinity(), u, v)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cmp;

    use crate::testbase::{
        tuple_to_point, tuplevec_small_strategy, TAabb3, TPoint3, TRay3, TVector3, TupleVec,
    };

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
            let (start_dist, end_dist) = ray.intersection_slice_for_aabb(&aabb);
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

            let (start_dist, end_dist) = ray.intersection_slice_for_aabb(&aabb);
            if aabb.contains(&ray.origin) {
                // ray inside of aabb
                assert!(start_dist < end_dist);
                assert!(start_dist >= 0.0);
            } else {
                // ray outside of aabb and doesn't intersect it
                assert!((start_dist, end_dist) == (-1.0, -1.0));
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

                if !(intersection_inside || close_to_border) {
                    println!("uvsum {}", uv_sum);
                    println!("intersects.0 {}", intersects.distance);
                    println!("intersects.1 (u) {}", intersects.u);
                    println!("intersects.2 (v) {}", intersects.v);
                    println!("u {}", u);
                    println!("v {}", v);
                }

                assert!(intersection_inside || close_to_border);
            }
        }
    }
}

impl<T: BHValue, const D: usize> IntersectsAabb<T, D> for Ray<T, D> {
    fn intersects_aabb(&self, aabb: &Aabb<T, D>) -> bool {
        self.intersects_aabb(aabb)
    }
}

#[cfg(all(feature = "bench", test))]
mod bench {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use test::{black_box, Bencher};

    use crate::testbase::{tuple_to_point, tuple_to_vector, TAabb3, TRay3, TupleVec};

    /// Generate a random deterministic `Ray`.
    fn random_ray(rng: &mut StdRng) -> TRay3 {
        let a = tuple_to_point(&rng.gen::<TupleVec>());
        let b = tuple_to_vector(&rng.gen::<TupleVec>());
        TRay3::new(a, b)
    }

    /// Generate a random deterministic `Aabb`.
    fn random_aabb(rng: &mut StdRng) -> TAabb3 {
        let a = tuple_to_point(&rng.gen::<TupleVec>());
        let b = tuple_to_point(&rng.gen::<TupleVec>());

        TAabb3::empty().grow(&a).grow(&b)
    }

    /// Generate the ray and boxes used for benchmarks.
    fn random_ray_and_boxes() -> (TRay3, Vec<TAabb3>) {
        let seed = [0; 32];
        let mut rng = StdRng::from_seed(seed);

        let ray = random_ray(&mut rng);
        let boxes = (0..1000).map(|_| random_aabb(&mut rng)).collect::<Vec<_>>();

        black_box((ray, boxes))
    }

    /// Benchmark for the optimized intersection algorithm.
    #[bench]
    fn bench_intersects_aabb(b: &mut Bencher) {
        let (ray, boxes) = random_ray_and_boxes();

        b.iter(|| {
            for aabb in &boxes {
                black_box(ray.intersects_aabb(aabb));
            }
        });
    }
}
