//! This module defines the `BoundingHierarchy` trait.

use aabb::Bounded;
use ray::Ray;

/// This trait defines an acceleration structure with space partitioning.
/// This structure is used to efficiently compute ray-scene intersections.
pub trait BoundingHierarchy {
    /// Creates a new [`BoundingHierarchy`] from the `shapes` slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use bvh::aabb::{AABB, Bounded};
    /// use bvh::bounding_hierarchy::BoundingHierarchy;
    /// use bvh::nalgebra::{Point3, Vector3};
    /// # fn create_bounded_shapes() -> Vec<AABB> {
    /// #     let mut shapes = Vec::new();
    /// #     let offset = Vector3::new(1.0, 1.0, 1.0);
    /// #     for i in 0..1000u32 {
    /// #         let position = Point3::new(i as f32, i as f32, i as f32);
    /// #         shapes.push(AABB::with_bounds(position - offset, position + offset));
    /// #     }
    /// #     shapes
    /// # }
    ///
    /// let shapes = create_bounded_shapes();
    /// // Construct a normal `BVH`.
    /// {
    ///     use bvh::bvh::BVH;
    ///     let bvh = BVH::build(&shapes);
    /// }
    ///
    /// // Or construct a `FlatBVH`.
    /// {
    ///     use bvh::flat_bvh::FlatBVH;
    ///     let bvh = FlatBVH::build(&shapes);
    /// }
    /// ```
    ///
    /// [`BoundingHierarchy`]: trait.BoundingHierarchy.html
    ///
    fn build<T: Bounded>(shapes: &[T]) -> Self;

    /// Traverses the [`BoundingHierarchy`] recursively.
    /// Returns a subset of `shapes`, in which the [`AABB`]s of the elements were hit by `ray`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bvh::aabb::{AABB, Bounded};
    /// use bvh::bounding_hierarchy::BoundingHierarchy;
    /// use bvh::bvh::BVH;
    /// use bvh::nalgebra::{Point3, Vector3};
    /// use bvh::ray::Ray;
    /// # fn create_bvh() -> (BVH, Vec<AABB>) {
    /// #     let mut shapes = Vec::new();
    /// #     let offset = Vector3::new(1.0, 1.0, 1.0);
    /// #     for i in 0..1000u32 {
    /// #         let position = Point3::new(i as f32, i as f32, i as f32);
    /// #         shapes.push(AABB::with_bounds(position - offset, position + offset));
    /// #     }
    /// #     let bvh = BVH::build(&shapes);
    /// #     (bvh, shapes)
    /// # }
    ///
    /// let (bvh, shapes) = create_bvh();
    ///
    /// let origin = Point3::new(0.0, 0.0, 0.0);
    /// let direction = Vector3::new(1.0, 0.0, 0.0);
    /// let ray = Ray::new(origin, direction);
    /// let hit_shapes = bvh.traverse(&ray, &shapes);
    /// ```
    ///
    /// [`BoundingHierarchy`]: trait.BoundingHierarchy.html
    /// [`AABB`]: ../aabb/struct.AABB.html
    ///
    fn traverse<'a, T: Bounded>(&'a self, ray: &Ray, shapes: &'a [T]) -> Vec<&T>;

    /// Prints the [`BoundingHierarchy`] in a tree-like visualization.
    ///
    /// [`BoundingHierarchy`]: ../bounding_hierarchy/trait.BoundingHierarchy.html
    ///
    fn pretty_print(&self) {}
}
