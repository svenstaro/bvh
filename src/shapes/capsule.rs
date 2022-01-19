//! This module defines Capsules and their intersection algorithms
use crate::{
    aabb::AABB, bounding_hierarchy::IntersectionAABB, utils::nearest_point_on_line, Point3, Real,
    Vector3,
};

/// Representation of a capsule
pub struct Capsule {
    /// Start point of the line segment for the capsule
    pub start: Point3,
    /// Radius of the capsule
    pub radius: Real,
    /// Direction of the capsule's line segment
    pub dir: Vector3,
    /// Length of the capsules line segment
    pub len: Real,
}

impl Capsule {
    /// Creates a capsule given a start and end point for the center line and the radius around it
    pub fn new(start: Point3, end: Point3, radius: Real) -> Capsule {
        let line = end - start;
        let dir = line.normalize();
        let len = line.length();

        Capsule {
            start,
            radius,
            dir,
            len,
        }
    }
}

impl IntersectionAABB for Capsule {
    fn intersects_aabb(&self, aabb: &AABB) -> bool {
        /*
        // Use Distance from closest point
        let mut point: Vector3 = self.start;
        let mut curr_d = 0.0;
        let max_sq = (self.len + self.radius) * (self.len + self.radius) ;
        let r_sq = self.radius * self.radius;

        loop {
            let x = aabb.closest_point(point);
            let d_sq = x.distance_squared(point);
            println!("{:?} closest={:?} d={:?}", point, x, d_sq.sqrt());
            if d_sq <= r_sq
            {
                return true;
            }
            if d_sq > max_sq || curr_d >= self.len
            {
                return false;
            }
            curr_d = (curr_d + d_sq.sqrt()).min(self.len);
            point = self.start + (curr_d * self.dir);
        }
        */
        let mut last = self.start;
        loop {
            let closest = &aabb.closest_point(last);
            let center = nearest_point_on_line(&self.start, &self.dir, self.len, closest);
            let sphere = crate::sphere::Sphere {
                center,
                radius: self.radius,
            };
            if sphere.intersects_aabb(aabb) {
                return true;
            }
            if last.distance_squared(center) < 0.0001 {
                return false;
            } else {
                last = center;
            }
        }
    }
}
