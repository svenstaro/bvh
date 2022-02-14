//! This module defines a Sphere and its intersection algorithms

use crate::{
    aabb::{Bounded, AABB},
    bounding_hierarchy::IntersectionAABB,
    ray::{Intersection, IntersectionRay, Ray},
    Point3, Real, Vector3, PI,
};

/// A representation of a Sphere
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde_impls", derive(serde::Serialize, serde::Deserialize))]
pub struct Sphere {
    /// Center of the sphere
    pub center: Point3,
    /// Radius of the sphere
    pub radius: Real,
}

impl Sphere {
    /// Creates a sphere centered on a given point with a radius
    pub fn new(center: Point3, radius: Real) -> Sphere {
        Sphere { center, radius }
    }
}

impl IntersectionAABB for Sphere {
    fn intersects_aabb(&self, aabb: &AABB) -> bool {
        let closest = aabb.closest_point(self.center);
        closest.distance_squared(self.center) <= self.radius * self.radius
    }
}

impl IntersectionRay for Sphere {
    fn intersects_ray(&self, ray: &Ray, t_min: Real, t_max: Real) -> Option<Intersection> {
        let oc = ray.origin - self.center;
        let a = ray.direction.length_squared();
        let half_b = oc.dot(ray.direction);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0. {
            return None;
        }

        let sqrtd = discriminant.sqrt();

        let mut toi = (-half_b - sqrtd) / a;
        if toi < t_min || t_max < toi {
            toi = (-half_b + sqrtd) / a;
            if toi < t_min || t_max < toi {
                return None;
            }
        }

        let hit = ray.at(toi);

        let out_norm = (hit - self.center) / self.radius;

        let theta = (-out_norm.y).acos();
        let phi = (-out_norm.z).atan2(out_norm.x) + PI;
        let u = phi / (2. * PI);
        let v = theta / PI;

        let (norm, back_face) = ray.face_normal(out_norm);
        Some(Intersection::new(toi, u, v, norm, back_face))
    }
}

impl Bounded for Sphere {
    fn aabb(&self) -> AABB {
        let min = self.center - Vector3::splat(self.radius);
        let max = self.center + Vector3::splat(self.radius);
        AABB::with_bounds(min, max)
    }
}
