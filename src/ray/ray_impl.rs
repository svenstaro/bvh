//! This module defines a Ray structure and intersection algorithms
//! for axis aligned bounding boxes and triangles.

use core::cmp::Ordering;
use nalgebra::{
    ClosedAddAssign, ClosedMulAssign, ClosedSubAssign, ComplexField, Point, SVector, SimdPartialOrd,
};
use num_traits::{Float, One, Zero};

use super::intersect_default::RayIntersection;
use crate::aabb::IntersectsAabb;
use crate::utils::{fast_max, has_nan};
use crate::{aabb::Aabb, bounding_hierarchy::BHValue};

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
    /// If there are no intersections, it returns `None`.
    pub fn intersection_slice_for_aabb(&self, aabb: &Aabb<T, D>) -> Option<(T, T)>
    where
        T: BHValue,
    {
        // Copied from the default `RayIntersection` implementation. TODO: abstract and add SIMD.
        let lbr = (aabb[0].coords - self.origin.coords).component_mul(&self.inv_direction);
        let rtr = (aabb[1].coords - self.origin.coords).component_mul(&self.inv_direction);

        if has_nan(&lbr) | has_nan(&rtr) {
            // Assumption: the ray is in the plane of an AABB face. Be consistent and
            // consider this a non-intersection. This avoids making the result depend
            // on which axis/axes have NaN (min/max in the code that follows are not
            // commutative).
            return None;
        }

        let (inf, sup) = lbr.inf_sup(&rtr);

        let tmin = fast_max(inf.max(), T::zero());
        let tmax = sup.min();

        if matches!(tmin.partial_cmp(&tmax), Some(Ordering::Greater) | None) {
            // tmin > tmax or either was NaN, meaning no intersection.
            return None;
        }

        Some((tmin, tmax))
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

impl<T: BHValue, const D: usize> IntersectsAabb<T, D> for Ray<T, D> {
    fn intersects_aabb(&self, aabb: &Aabb<T, D>) -> bool {
        self.intersects_aabb(aabb)
    }
}
