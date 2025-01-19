//! Contains the `PointDistance` trait used for querying the distance to a point to a bvh.
use crate::bounding_hierarchy::BHValue;

/// A trait implemented by shapes that can be queried for their distance to a point.
///
/// Used for the `Bvh::nearest_to` method that returns the nearest shape to a point.
pub trait PointDistance<T: BHValue, const D: usize> {
    /// Returns the squared distance from this point to the Shape.
    fn distance_squared(&self, query_point: nalgebra::Point<T, D>) -> T;
}
