//! Axis Aligned Bounding Boxes.

use nalgebra::{Point3, Vector3};
use std::f32;
use std::ops::Index;

/// AABB struct.
#[derive(Debug, Copy, Clone)]
pub struct AABB {
    // Minimum coordinates
    pub min: Point3<f32>,

    // Maximum coordinates
    pub max: Point3<f32>,
}

/// A trait implemented by things which can be bounded by an `AABB`.
pub trait Bounded {
    fn aabb(&self) -> AABB;
}

impl AABB {
    /// Creates a new `AABB` with the given bounds.
    ///
    /// # Examples
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate bvh;
    /// use bvh::aabb::AABB;
    /// use nalgebra::Point3;
    ///
    /// # fn main() {
    /// let aabb = AABB::with_bounds(Point3::new(-1.0,-1.0,-1.0), Point3::new(1.0,1.0,1.0));
    /// assert_eq!(aabb.min.x, -1.0);
    /// assert_eq!(aabb.max.z, 1.0);
    /// # }
    /// ```
    pub fn with_bounds(min: Point3<f32>, max: Point3<f32>) -> AABB {
        AABB {
            min: min,
            max: max,
        }
    }

    /// Creates a new empty `AABB`.
    ///
    /// # Examples
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate bvh;
    /// # extern crate rand;
    /// use bvh::aabb::AABB;
    /// use nalgebra::Point3;
    ///
    /// # fn main() {
    /// let aabb = AABB::empty();
    /// let min = &aabb.min;
    /// let max = &aabb.max;
    ///
    /// // For any point
    /// let x = rand::random();
    /// let y = rand::random();
    /// let z = rand::random();
    ///
    /// // An empty AABB should not contain it
    /// assert!(x < min.x && y < min.y && z < min.z);
    /// assert!(max.x < x && max.y < y && max.z < z);
    /// # }
    /// ```
    pub fn empty() -> AABB {
        AABB {
            min: Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    /// Returns true if the `Point3` is inside the `AABB`.
    ///
    /// # Examples
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate bvh;
    /// use bvh::aabb::AABB;
    /// use nalgebra::Point3;
    ///
    /// # fn main() {
    /// let aabb = AABB::with_bounds(Point3::new(-1.0,-1.0,-1.0), Point3::new(1.0,1.0,1.0));
    /// let point_inside = Point3::new(0.125,-0.25,0.5);
    /// let point_outside = Point3::new(1.0,-2.0,4.0);
    ///
    /// assert!(aabb.contains(&point_inside));
    /// assert!(!aabb.contains(&point_outside));
    /// # }
    /// ```
    pub fn contains(&self, p: &Point3<f32>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y &&
        p.z >= self.min.z && p.z <= self.max.z
    }

    /// Returns true if the `Point3` is approximately inside the `AABB`
    /// with respect to some `epsilon`.
    ///
    /// # Examples
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate bvh;
    /// use bvh::aabb::AABB;
    /// use nalgebra::Point3;
    ///
    /// # fn main() {
    /// let aabb = AABB::with_bounds(Point3::new(-1.0,-1.0,-1.0), Point3::new(1.0,1.0,1.0));
    /// let point_barely_inside = Point3::new(1.0000001,-1.0000001,1.000000001);
    /// let point_outside = Point3::new(1.0,-2.0,4.0);
    /// const EPSILON: f32 = 0.000001;
    ///
    /// assert!(aabb.approx_contains_eps(&point_barely_inside, EPSILON));
    /// assert!(!aabb.approx_contains_eps(&point_outside, EPSILON));
    /// # }
    /// ```
    pub fn approx_contains_eps(&self, p: &Point3<f32>, epsilon: f32) -> bool {
        (p.x - self.min.x) > -epsilon && (p.x - self.max.x) < epsilon &&
        (p.y - self.min.y) > -epsilon && (p.y - self.max.y) < epsilon &&
        (p.z - self.min.z) > -epsilon && (p.z - self.max.z) < epsilon
    }

    /// Returns a new minimal `AABB` which contains both this `AABB` and `other`.
    ///
    /// # Examples
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate bvh;
    /// use bvh::aabb::AABB;
    /// use nalgebra::Point3;
    ///
    /// # fn main() {
    /// let aabb1 = AABB::with_bounds(Point3::new(-101.0,0.0,0.0), Point3::new(-100.0,1.0,1.0));
    /// let aabb2 = AABB::with_bounds(Point3::new(100.0,0.0,0.0), Point3::new(101.0,1.0,1.0));
    /// let union = aabb1.union(&aabb2);
    ///
    /// let point_inside_aabb1 = Point3::new(-100.5,0.5,0.5);
    /// let point_inside_aabb2 = Point3::new(100.5,0.5,0.5);
    /// let point_inside_union = Point3::new(0.0,0.5,0.5);
    ///
    /// # assert!(aabb1.contains(&point_inside_aabb1));
    /// # assert!(!aabb1.contains(&point_inside_aabb2));
    /// # assert!(!aabb1.contains(&point_inside_union));
    /// #
    /// # assert!(!aabb2.contains(&point_inside_aabb1));
    /// # assert!(aabb2.contains(&point_inside_aabb2));
    /// # assert!(!aabb2.contains(&point_inside_union));
    ///
    /// assert!(union.contains(&point_inside_aabb1));
    /// assert!(union.contains(&point_inside_aabb2));
    /// assert!(union.contains(&point_inside_union));
    /// # }
    /// ```
    pub fn union(&self, other: &AABB) -> AABB {
        AABB::with_bounds(Point3::new(self.min.x.min(other.min.x),
                                      self.min.y.min(other.min.y),
                                      self.min.z.min(other.min.z)),
                          Point3::new(self.max.x.max(other.max.x),
                                      self.max.y.max(other.max.y),
                                      self.max.z.max(other.max.z)))
    }

    /// Returns a new minimal `AABB` which contains both this `AABB` and `other`.
    ///
    /// # Examples
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate bvh;
    /// use bvh::aabb::AABB;
    /// use nalgebra::Point3;
    ///
    /// # fn main() {
    /// let point1 = Point3::new(0.0,0.0,0.0);
    /// let point2 = Point3::new(1.0,1.0,1.0);
    /// let point3 = Point3::new(2.0,2.0,2.0);
    ///
    /// let aabb = AABB::empty();
    /// assert!(!aabb.contains(&point1));
    ///
    /// let aabb1 = aabb.grow(&point1);
    /// assert!(aabb1.contains(&point1));
    ///
    /// let aabb2 = aabb.grow(&point2);
    /// assert!(aabb2.contains(&point2));
    /// assert!(!aabb2.contains(&point3));
    /// # }
    /// ```
    pub fn grow(&self, other: &Point3<f32>) -> AABB {
        AABB::with_bounds(Point3::new(self.min.x.min(other.x),
                                      self.min.y.min(other.y),
                                      self.min.z.min(other.z)),
                          Point3::new(self.max.x.max(other.x),
                                      self.max.y.max(other.y),
                                      self.max.z.max(other.z)))
    }

    /// Returns a new minimal `AABB` which contains both this `AABB` and `other`.
    pub fn union_bounded<T: Bounded>(&self, other: &T) -> AABB {
        self.union(&other.aabb())
    }

    /// Returns the size of this `AABB` in all three dimensions.
    ///
    /// # Examples
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate bvh;
    /// use bvh::aabb::AABB;
    /// use nalgebra::Point3;
    ///
    /// # fn main() {
    /// let aabb = AABB::with_bounds(Point3::new(-1.0,-1.0,-1.0), Point3::new(1.0,1.0,1.0));
    /// let size = aabb.size();
    /// assert!(size.x == 2.0 && size.y == 2.0 && size.z == 2.0);
    /// # }
    /// ```
    pub fn size(&self) -> Vector3<f32> {
        self.max - self.min
    }

    /// Returns the center point of the `AABB`.
    ///
    /// # Examples
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate bvh;
    /// use bvh::aabb::AABB;
    /// use nalgebra::Point3;
    ///
    /// # fn main() {
    /// let min = Point3::new(41.0,41.0,41.0);
    /// let max = Point3::new(43.0,43.0,43.0);
    ///
    /// let aabb = AABB::with_bounds(min, max);
    /// let center = aabb.center();
    /// assert!(center.x == 42.0 && center.y == 42.0 && center.z == 42.0);
    /// # }
    /// ```
    pub fn center(&self) -> Point3<f32> {
        self.min + (self.size() / 2.0)
    }

    /// Returns the total surface area of this `AABB`.
    ///
    /// # Examples
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate bvh;
    /// use bvh::aabb::AABB;
    /// use nalgebra::Point3;
    ///
    /// # fn main() {
    /// let min = Point3::new(41.0,41.0,41.0);
    /// let max = Point3::new(43.0,43.0,43.0);
    ///
    /// let aabb = AABB::with_bounds(min, max);
    /// let surface_area = aabb.surface_area();
    /// assert!(surface_area == 24.0);
    /// # }
    /// ```
    pub fn surface_area(&self) -> f32 {
        let size = self.size();
        2.0 * (size.x * size.y + size.x * size.z + size.y * size.z)
    }

    /// Returns the volume of this `AABB`.
    ///
    /// # Examples
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate bvh;
    /// use bvh::aabb::AABB;
    /// use nalgebra::Point3;
    ///
    /// # fn main() {
    /// let min = Point3::new(41.0,41.0,41.0);
    /// let max = Point3::new(43.0,43.0,43.0);
    ///
    /// let aabb = AABB::with_bounds(min, max);
    /// let volume = aabb.volume();
    /// assert!(volume == 8.0);
    /// # }
    /// ```
    pub fn volume(&self) -> f32 {
        let size = self.size();
        size.x * size.y * size.z
    }
}

/// Make `AABB`s indexable. `aabb[0]` gives a reference to the minimum bound.
/// All other indices return a reference to the maximum bound.
impl Index<usize> for AABB {
    type Output = Point3<f32>;

    fn index(&self, index: usize) -> &Point3<f32> {
        if index == 0 { &self.min } else { &self.max }
    }
}

/// Implementation of `Bounded` for single points.
impl Bounded for Point3<f32> {
    fn aabb(&self) -> AABB {
        AABB::with_bounds(*self, *self)
    }
}

#[cfg(test)]
mod tests {
    use aabb::AABB;
    use nalgebra::Point3;

    type TupleVec = (f32, f32, f32);

    fn tuple_to_point(tpl: &TupleVec) -> Point3<f32> {
        Point3::new(tpl.0, tpl.1, tpl.2)
    }

    /// Test whether an empty `AABB` does not contains anything.
    quickcheck!{
        fn test_empty_contains_nothing(tpl: TupleVec) -> bool {
            // Define a random Point
            let p = tuple_to_point(&tpl);

            // Create an empty AABB
            let aabb = AABB::empty();

            // It should not contain anything
            !aabb.contains(&p)
        }
    }

    /// Test whether an AABB always contains its center.
    quickcheck!{
        fn test_aabb_contains_center(a: TupleVec, b: TupleVec) -> bool {
            // Define two points which will be the corners of the `AABB`
            let p1 = tuple_to_point(&a);
            let p2 = tuple_to_point(&b);

            // Span the `AABB`
            let aabb = AABB::empty().grow(&p1).union_bounded(&p2);

            // Its center should be inside the `AABB`
            aabb.contains(&aabb.center())
        }
    }

    /// Test whether the union of two point-sets contains all the points.
    quickcheck!{
        fn test_join_two_aabbs(a: (TupleVec, TupleVec, TupleVec, TupleVec, TupleVec),
                               b: (TupleVec, TupleVec, TupleVec, TupleVec, TupleVec))
                               -> bool {
            // Define an array of ten points
            let points = [a.0, a.1, a.2, a.3, a.4, b.0, b.1, b.2, b.3, b.4];

            // Convert these points to `Point3`
            let points = points.iter().map(tuple_to_point).collect::<Vec<Point3<f32>>>();

            // Create two `AABB`s. One spanned the first five points,
            // the other by the last five points
            let aabb1 = points.iter().take(5).fold(AABB::empty(), |aabb, point| aabb.grow(&point));
            let aabb2 = points.iter().skip(5).fold(AABB::empty(), |aabb, point| aabb.grow(&point));

            // The `AABB`s should contain the points by which they are spanned
            let aabb1_contains_init_five = points.iter()
                .take(5)
                .fold(true, |b, point| b && aabb1.contains(&point));
            let aabb2_contains_last_five = points.iter()
                .skip(5)
                .fold(true, |b, point| b && aabb2.contains(&point));

            // Build the union of the two `AABB`s
            let aabbu = aabb1.union(&aabb2);

            // The union should contain all points
            let aabbu_contains_all = points.iter()
                .fold(true, |b, point| b && aabbu.contains(&point));

            // Return the three properties
            aabb1_contains_init_five && aabb2_contains_last_five && aabbu_contains_all
        }
    }
}
