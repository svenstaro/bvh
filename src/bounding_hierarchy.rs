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
    /// # use bvh::bounding_hierarchy::BHShape;
    /// #
    /// # impl BHShape for AABB {
    /// #     fn set_bh_node_index(&mut self, index: usize) { }
    /// #     fn bh_node_index(&self) -> usize { 0 }
    /// # }
    /// #
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
    /// let mut shapes = create_bounded_shapes();
    /// // Construct a normal `BVH`.
    /// {
    ///     use bvh::bvh::BVH;
    ///     let bvh = BVH::build(&mut shapes);
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
    fn build<Shape: BHShape>(shapes: &mut [Shape]) -> Self;

    /// Traverses the [`BoundingHierarchy`].
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
    /// # use bvh::bounding_hierarchy::BHShape;
    /// #
    /// # impl BHShape for AABB {
    /// #     fn set_bh_node_index(&mut self, index: usize) { }
    /// #     fn bh_node_index(&self) -> usize { 0 }
    /// # }
    /// #
    /// # fn create_bvh() -> (BVH, Vec<AABB>) {
    /// #     let mut shapes = Vec::new();
    /// #     let offset = Vector3::new(1.0, 1.0, 1.0);
    /// #     for i in 0..1000u32 {
    /// #         let position = Point3::new(i as f32, i as f32, i as f32);
    /// #         shapes.push(AABB::with_bounds(position - offset, position + offset));
    /// #     }
    /// #     let bvh = BVH::build(&mut shapes);
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
    fn traverse<'a, Shape: BHShape>(&'a self, ray: &Ray, shapes: &'a [Shape]) -> Vec<&Shape>;

    /// Prints the [`BoundingHierarchy`] in a tree-like visualization.
    ///
    /// [`BoundingHierarchy`]: ../bounding_hierarchy/trait.BoundingHierarchy.html
    ///
    fn pretty_print(&self) {}
}

/// Describes a shape as referenced by a [`BoundingHierarchy`] leaf node.
/// Knows the index of the node in the [`BoundingHierarchy`] it is in.
///
/// [`BoundingHierarchy`]: struct.BoundingHierarchy.html
///
pub trait BHShape: Bounded {
    /// Sets the index of the referenced [`BoundingHierarchy`] node.
    ///
    /// [`BoundingHierarchy`]: struct.BoundingHierarchy.html
    ///
    fn set_bh_node_index(&mut self, usize);

    /// Gets the index of the referenced [`BoundingHierarchy`] node.
    ///
    /// [`BoundingHierarchy`]: struct.BoundingHierarchy.html
    ///
    fn bh_node_index(&self) -> usize;
}
