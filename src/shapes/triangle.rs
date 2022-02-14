//! This module defines a Triangle and its intersection algorithms

use crate::aabb::{Bounded, AABB};
use crate::bounding_hierarchy::IntersectionAABB;
use crate::shapes::ray::{Intersection, IntersectionRay, Ray};
use crate::{Point3, Real, Vector3};

/// A triangle struct. Instance of a more complex `Bounded` primitive.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde_impls", derive(serde::Serialize, serde::Deserialize))]
pub struct Triangle {
    /// First point on the triangle
    pub a: Point3,
    /// Second point on the triangle
    pub b: Point3,
    /// Third point on the triangle
    pub c: Point3,
}

impl Triangle {
    /// Creates a new triangle given a counter clockwise set of points
    pub fn new(a: Point3, b: Point3, c: Point3) -> Triangle {
        Triangle { a, b, c }
    }
}

impl Bounded for Triangle {
    fn aabb(&self) -> AABB {
        AABB::empty().grow(&self.a).grow(&self.b).grow(&self.c)
    }
}

impl IntersectionRay for Triangle {
    fn intersects_ray(&self, ray: &Ray, t_min: Real, t_max: Real) -> Option<Intersection> {
        let inter = ray.intersects_triangle(&self.a, &self.b, &self.c);
        if inter.distance <= t_max && inter.distance >= t_min {
            Some(inter)
        } else {
            None
        }
    }
}

impl IntersectionAABB for Triangle {
    fn intersects_aabb(&self, aabb: &AABB) -> bool {
        let c = aabb.center();
        let extents = ((aabb.max - aabb.min) / 2.).to_array();

        let verts = [self.a - c, self.b - c, self.c - c];

        let lines = [
            verts[1] - verts[0],
            verts[2] - verts[1],
            verts[0] - verts[2],
        ];

        let normals = [Vector3::X, Vector3::Y, Vector3::Z];

        for u in normals.iter() {
            for f in lines.iter() {
                let axis = u.cross(*f);
                if !separating_axis_test(&verts, &normals, &extents, axis) {
                    return false;
                }
            }
        }

        for a in normals {
            if !separating_axis_test(&verts, &normals, &extents, a) {
                return false;
            }
        }

        let triangle_normal = lines[0].cross(lines[1]);
        if !separating_axis_test(&verts, &normals, &extents, triangle_normal) {
            return false;
        }

        true
    }
}

fn separating_axis_test(
    verts: &[Vector3; 3],
    normals: &[Vector3; 3],
    extents: &[Real; 3],
    axis: Vector3,
) -> bool {
    let projection = Vector3::new(verts[0].dot(axis), verts[1].dot(axis), verts[2].dot(axis));

    let r: Real = extents
        .iter()
        .zip(normals.iter())
        .map(|(&e, u)| e * (u.dot(axis)))
        .sum();

    let max = projection.max_element();
    let min = projection.min_element();
    (-max).max(min) <= r
}
