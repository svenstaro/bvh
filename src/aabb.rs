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
    pub fn new(min: Point3<f32>, max: Point3<f32>) -> AABB {
        AABB {
            min: min,
            max: max,
        }
    }

    /// Creates a new empty `AABB`.
    pub fn empty() -> AABB {
        AABB {
            min: Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    /// Returns true if the `Point3` is inside the `AABB`.
    pub fn contains(&self, p: &Point3<f32>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y &&
        p.z >= self.min.z && p.z <= self.max.z
    }

    /// Returns true if the `Point3` is approximately inside the `AABB`
    /// with respect to some `epsilon`.
    pub fn approx_contains_eps(&self, p: &Point3<f32>, epsilon: f32) -> bool {
        (p.x - self.min.x) > -epsilon && (p.x - self.max.x) < epsilon &&
        (p.y - self.min.y) > -epsilon && (p.y - self.max.y) < epsilon &&
        (p.z - self.min.z) > -epsilon && (p.z - self.max.z) < epsilon
    }

    /// Returns a new minimal `AABB` which contains both this `AABB` and `other`.
    pub fn union_aabb(&self, other: &AABB) -> AABB {
        AABB::new(Point3::new(self.min.x.min(other.min.x),
                              self.min.y.min(other.min.y),
                              self.min.z.min(other.min.z)),
                  Point3::new(self.max.x.max(other.max.x),
                              self.max.y.max(other.max.y),
                              self.max.z.max(other.max.z)))
    }

    /// Returns a new minimal `AABB` which contains both this `AABB` and `other`.
    pub fn union_point(&self, other: &Point3<f32>) -> AABB {
        AABB::new(Point3::new(self.min.x.min(other.x),
                              self.min.y.min(other.y),
                              self.min.z.min(other.z)),
                  Point3::new(self.max.x.max(other.x),
                              self.max.y.max(other.y),
                              self.max.z.max(other.z)))
    }

    /// Returns a new minimal `AABB` which contains both this `AABB` and `other`.
    pub fn union_bounded<T: Bounded>(&self, other: &T) -> AABB {
        self.union_aabb(&other.aabb())
    }

    /// Returns the size of this `AABB` in all three dimensions.
    pub fn size(&self) -> Vector3<f32> {
        self.max - self.min
    }

    /// Returns the center point of the `AABB`.
    pub fn center(&self) -> Point3<f32> {
        self.min + (self.size() / 2.0)
    }

    /// Returns the total surface area of this `AABB`.
    pub fn surface_area(&self) -> f32 {
        let size = self.size();
        2.0 * size.x * size.y + size.x * size.z + size.y * size.z
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
        AABB::new(*self, *self)
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
            let p = tuple_to_point(&tpl);
            let aabb = AABB::empty();
            !aabb.contains(&p)
        }
    }

    /// Test whether an AABB always contains its center.
    quickcheck!{
        fn test_aabb_contains_center(a: TupleVec, b: TupleVec) -> bool {
            let p1 = tuple_to_point(&a);
            let p2 = tuple_to_point(&b);
            let aabb = AABB::empty().union_point(&p1).union_bounded(&p2);
            aabb.contains(&aabb.center())
        }
    }

    /// Test whether the union of two point-sets contains all the points.
    quickcheck!{
        fn test_join_two_aabbs(a: (TupleVec, TupleVec, TupleVec, TupleVec, TupleVec),
                               b: (TupleVec, TupleVec, TupleVec, TupleVec, TupleVec))
                               -> bool {
            let points = [a.0, a.1, a.2, a.3, a.4, b.0, b.1, b.2, b.3, b.4];
            let points = points.iter().map(tuple_to_point).collect::<Vec<Point3<f32>>>();
            let aabb1 =
                points.iter().take(5).fold(AABB::empty(), |aabb, point| aabb.union_point(&point));
            let aabb2 =
                points.iter().skip(5).fold(AABB::empty(), |aabb, point| aabb.union_point(&point));

            let aabb1_contains_init_five = points.iter()
                .take(5)
                .fold(true, |b, point| b && aabb1.contains(&point));
            let aabb2_contains_last_five = points.iter()
                .skip(5)
                .fold(true, |b, point| b && aabb2.contains(&point));

            let aabbu = aabb1.union_aabb(&aabb2);
            let aabbu_contains_all = points.iter()
                .fold(true, |b, point| b && aabbu.contains(&point));

            aabb1_contains_init_five && aabb2_contains_last_five && aabbu_contains_all
        }
    }

    // Rotates the camera to point in the direction [-1, 1, -1] and moves it dist*sqrt3 to the front.
    // The camera should now be at position [-dist, dist, -dist].
    // quickcheck!{
    // fn look_and_move_front_top_left(dist: f32) -> bool {
    // let mut camera = Camera::new(Point3::origin());
    //
}
