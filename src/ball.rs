//! Balls, including circles and spheres.

use crate::{
    aabb::{Aabb, IntersectsAabb},
    bounding_hierarchy::BHValue,
};
use nalgebra::Point;

/// In 2D, a circle. In 3D, a sphere. This can be used for traversing BVH's.
pub struct Ball<T: BHValue, const D: usize> {
    /// The center of the ball.
    pub center: Point<T, D>,
    /// The radius of the ball.
    pub radius: T,
}

impl<T: BHValue, const D: usize> Ball<T, D> {
    /// Creates a [`Ball`] with the given `center` and `radius`.
    ///
    /// # Panics
    /// Panics, in debug mode, if the radius is negative.
    ///
    /// # Examples
    /// ```
    /// use bvh::ball::Ball;
    /// use nalgebra::Point3;
    ///
    /// let ball = Ball::new(Point3::new(1.0, 1.0, 1.0), 1.0);
    /// assert_eq!(ball.center, Point3::new(1.0, 1.0, 1.0));
    /// assert_eq!(ball.radius, 1.0)
    /// ```
    ///
    /// [`Ball`]: struct.Ball.html
    pub fn new(center: Point<T, D>, radius: T) -> Self {
        debug_assert!(radius >= T::from_f32(0.0).unwrap());
        Self { center, radius }
    }

    /// Returns true if this [`Ball`] contains the [`Point`].
    ///
    /// # Examples
    /// ```
    /// use bvh::ball::Ball;
    /// use nalgebra::Point3;
    ///
    /// let ball = Ball::new(Point3::new(1.0, 1.0, 1.0), 1.0);
    /// let point = Point3::new(1.25, 1.25, 1.25);
    ///
    /// assert!(ball.contains(&point));
    /// ```
    ///
    /// [`Ball`]: struct.Ball.html
    pub fn contains(&self, point: &Point<T, D>) -> bool {
        let mut distance_squared = T::zero();
        for i in 0..D {
            distance_squared += (point[i] - self.center[i]).powi(2);
        }
        // Squaring the RHS is faster than computing the square root of the LHS.
        distance_squared <= self.radius.powi(2)
    }

    /// Returns true if this [`Ball`] intersects the [`Aabb`].
    ///
    /// # Examples
    /// ```
    /// use bvh::{aabb::Aabb, ball::Ball};
    /// use nalgebra::Point3;
    ///
    /// let ball = Ball::new(Point3::new(1.0, 1.0, 1.0), 1.0);
    /// let aabb = Aabb::with_bounds(Point3::new(1.25, 1.25, 1.25), Point3::new(3.0, 3.0, 3.0));
    ///
    /// assert!(ball.intersects_aabb(&aabb));
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    /// [`Ball`]: struct.Ball.html
    pub fn intersects_aabb(&self, aabb: &Aabb<T, D>) -> bool {
        // https://gamemath.com/book/geomtests.html#intersection_sphere_aabb
        // Finding the point in/on the AABB that is closest to the ball's center or,
        // more specifically, find the squared distance between that point and the
        // ball's center.
        let mut distance_squared = T::zero();
        for i in 0..D {
            let closest_on_aabb = self.center[i].clamp(aabb.min[i], aabb.max[i]);
            distance_squared += (closest_on_aabb - self.center[i]).powi(2);
        }

        // Then test if that point is in/on the ball. Squaring the RHS is faster than computing
        // the square root of the LHS.
        distance_squared <= self.radius.powi(2)
    }
}

impl<T: BHValue, const D: usize> IntersectsAabb<T, D> for Ball<T, D> {
    fn intersects_aabb(&self, aabb: &Aabb<T, D>) -> bool {
        self.intersects_aabb(aabb)
    }
}

#[cfg(test)]
mod tests {
    use super::Ball;
    use crate::testbase::TPoint3;

    #[test]
    fn ball_contains() {
        let ball = Ball::new(TPoint3::new(3.0, 4.0, 5.0), 1.5);

        // Ball should contain its own center.
        assert!(ball.contains(&ball.center));

        // Test some manually-selected points.
        let just_inside = TPoint3::new(3.04605, 3.23758, 3.81607);
        let just_outside = TPoint3::new(3.06066, 3.15813, 3.70917);
        assert!(ball.contains(&just_inside));
        assert!(!ball.contains(&just_outside));
    }
}
