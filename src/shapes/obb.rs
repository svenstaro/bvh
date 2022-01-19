//! This module defines an Oriented Bounding Box and its intersection properties
use crate::{aabb::AABB, bounding_hierarchy::IntersectionAABB, Mat4, Quat, Vector3};

/// Represents a box that can be rotated in any direction
pub struct OBB {
    /// Orientation of the OBB
    pub orientation: Quat,
    /// Extents of the box before being transformed by the orientation
    pub extents: Vector3,
    /// Center of the box
    pub center: Vector3,
}

impl IntersectionAABB for OBB {
    fn intersects_aabb(&self, aabb: &AABB) -> bool {
        let half_a = self.extents;
        let half_b = (aabb.max - aabb.min) * 0.5;
        let value = (aabb.max + aabb.min) * 0.5;
        let translation = self.orientation * (value - self.center);
        let mat = Mat4::from_rotation_translation(self.orientation, translation.into());

        let vec_1 = Vector3::new(
            translation.x.abs(),
            translation.y.abs(),
            translation.z.abs(),
        );
        let right = right(mat);
        let up = up(mat);
        let backward = back(mat);
        let vec_2 = right * half_b.x;
        let vec_3 = up * half_b.y;
        let vec_4 = backward * half_b.z;
        let num = vec_2.x.abs() + vec_3.x.abs() + vec_4.x.abs();
        let num2 = vec_2.y.abs() + vec_3.y.abs() + vec_4.y.abs();
        let num3 = vec_2.z.abs() + vec_3.z.abs() + vec_4.z.abs();
        if vec_1.x + num <= half_a.x && vec_1.y + num2 <= half_a.y && vec_1.z + num3 <= half_a.z {
            // Contained
            return true;
        }
        if vec_1.x > half_a.x + vec_2.x.abs() + vec_3.x.abs() + vec_4.x.abs() {
            return false;
        }
        if vec_1.y > half_a.y + vec_2.y.abs() + vec_3.y.abs() + vec_4.y.abs() {
            return false;
        }
        if vec_1.z > half_a.z + vec_2.z.abs() + vec_3.z.abs() + vec_4.z.abs() {
            return false;
        }
        if translation.dot(right.abs())
            > half_a.x * right.x.abs()
                + half_a.y * right.y.abs()
                + half_a.z * right.z.abs()
                + half_b.x
        {
            return false;
        }
        if translation.dot(up.abs())
            > half_a.x * up.x.abs() + half_a.y * up.y.abs() + half_a.z * up.z.abs() + half_b.y
        {
            return false;
        }
        if translation.dot(backward.abs())
            > half_a.x * backward.x.abs()
                + half_a.y * backward.y.abs()
                + half_a.z * backward.z.abs()
                + half_b.z
        {
            return false;
        }
        let mut vec_5 = Vector3::new(0.0, -right.z, right.y);
        if translation.dot(vec_5.abs())
            > half_a.y * vec_5.y.abs()
                + half_a.z * vec_5.z.abs()
                + vec_5.dot(vec_3.abs())
                + vec_5.dot(vec_4.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(0.0, -up.z, up.y);
        if translation.dot(vec_5.abs())
            > half_a.y * vec_5.y.abs()
                + half_a.z * vec_5.z.abs()
                + vec_5.dot(vec_4.abs())
                + vec_5.dot(vec_2.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(0.0, -backward.z, backward.y);
        if translation.dot(vec_5.abs())
            > half_a.y * vec_5.y.abs()
                + half_a.z * vec_5.z.abs()
                + vec_5.dot(vec_2.abs())
                + vec_5.dot(vec_3.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(right.z, 0.0, -right.x);
        if translation.dot(vec_5.abs())
            > half_a.z * vec_5.z.abs()
                + half_a.x * vec_5.x.abs()
                + vec_5.dot(vec_3.abs())
                + vec_5.dot(vec_4.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(up.z, 0.0, -up.x);
        if translation.dot(vec_5.abs())
            > half_a.z * vec_5.z.abs()
                + half_a.x * vec_5.x.abs()
                + vec_5.dot(vec_4.abs())
                + vec_5.dot(vec_2.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(backward.z, 0.0, -backward.x);
        if translation.dot(vec_5.abs())
            > half_a.z * vec_5.z.abs()
                + half_a.x * vec_5.x.abs()
                + vec_5.dot(vec_2.abs())
                + vec_5.dot(vec_3.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(-right.y, right.x, 0.0);
        if translation.dot(vec_5.abs())
            > half_a.x * vec_5.x.abs()
                + half_a.y * vec_5.y.abs()
                + vec_5.dot(vec_3.abs())
                + vec_5.dot(vec_4.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(-up.y, up.x, 0.0);
        if translation.dot(vec_5.abs())
            > half_a.x * vec_5.x.abs()
                + half_a.y * vec_5.y.abs()
                + vec_5.dot(vec_4.abs())
                + vec_5.dot(vec_2.abs())
        {
            return false;
        }
        vec_5 = Vector3::new(-backward.y, backward.x, 0.0);
        if translation.dot(vec_5.abs())
            > half_a.x * vec_5.x.abs()
                + half_a.y * vec_5.y.abs()
                + vec_5.dot(vec_2.abs())
                + vec_5.dot(vec_3.abs())
        {
            return false;
        }
        // Intersection
        return true;
    }
}

fn right(matrix: Mat4) -> Vector3 {
    matrix.row(0).truncate().into()
}

fn up(matrix: Mat4) -> Vector3 {
    matrix.row(1).truncate().into()
}

fn back(matrix: Mat4) -> Vector3 {
    matrix.row(2).truncate().into()
}

fn translation(matrix: Mat4) -> Vector3 {
    matrix.row(3).truncate().into()
}
