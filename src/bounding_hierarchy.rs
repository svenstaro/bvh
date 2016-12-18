//! This module defines the `BoundingHierarchy` trait.

use aabb::Bounded;
use ray::Ray;

/// This trait defines an acceleration structure with space partitioning.
/// This structure is used to efficiently compute ray-scene intersections.
pub trait BoundingHierarchy {
    /// Creates a new [`BoundingHierarchy`] from the `shapes` slice.
    ///
    /// [`BoundingHierarchy`]: trait.BoundingHierarchy.html
    ///
    /// # Examples
    ///
    /// // TODO example
    ///
    fn build<T: Bounded>(shapes: &[T]) -> Self;

    /// Traverses the [`BoundingHierarchy`] recursively.
    /// Returns a subset of `shapes`, in which the [`AABB`]s of the elements were hit by `ray`.
    ///
    /// [`BoundingHierarchy`]: trait.BoundingHierarchy.html
    /// [`AABB`]: ../aabb/struct.AABB.html
    ///
    /// # Examples
    ///
    /// // TODO example
    ///
    fn traverse<'a, T: Bounded>(&'a self, ray: &Ray, shapes: &'a [T]) -> Vec<&T>;

    /// Prints the [`BoundingHierarchy`] in a tree-like visualization.
    ///
    /// [`BoundingHierarchy`]: ../bounding_hierarchy/trait.BoundingHierarchy.html
    ///
    fn pretty_print(&self) {}
}
