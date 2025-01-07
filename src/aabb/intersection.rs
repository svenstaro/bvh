use nalgebra::Point;

use crate::{aabb::Aabb, bounding_hierarchy::BHValue};

/// A trait implemented by things that may or may not intersect an AABB and, by extension,
/// things that can be used to traverse a BVH.
pub trait IntersectsAabb<T: BHValue, const D: usize> {
    /// Returns whether this object intersects an [`Aabb`].
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::{Aabb, IntersectsAabb};
    /// use nalgebra::Point3;
    ///
    /// struct XyPlane;
    ///
    /// impl IntersectsAabb<f32,3> for XyPlane {
    ///     fn intersects_aabb(&self, aabb: &Aabb<f32,3>) -> bool {
    ///         aabb.min[2] <= 0.0 && aabb.max[2] >= 0.0
    ///     }
    /// }
    ///
    /// let xy_plane = XyPlane;
    /// let aabb = Aabb::with_bounds(Point3::new(-1.0,-1.0,-1.0), Point3::new(1.0,1.0,1.0));
    /// assert!(xy_plane.intersects_aabb(&aabb));
    /// ```
    ///
    /// [`Aabb`]: struct.Aabb.html
    ///
    fn intersects_aabb(&self, aabb: &Aabb<T, D>) -> bool;
}

impl<T: BHValue, const D: usize> IntersectsAabb<T, D> for Aabb<T, D> {
    fn intersects_aabb(&self, aabb: &Aabb<T, D>) -> bool {
        for i in 0..D {
            if self.max[i] < aabb.min[i] || aabb.max[i] < self.min[i] {
                return false;
            }
        }
        true
    }
}

impl<T: BHValue, const D: usize> IntersectsAabb<T, D> for Point<T, D> {
    fn intersects_aabb(&self, aabb: &Aabb<T, D>) -> bool {
        aabb.contains(self)
    }
}
