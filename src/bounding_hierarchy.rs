//! This module defines the [`BoundingHierarchy`] trait.

use alloc::{boxed::Box, vec::Vec};
use core::fmt;

use nalgebra::{
    ClosedAddAssign, ClosedDivAssign, ClosedMulAssign, ClosedSubAssign, Point, Scalar,
    SimdPartialOrd,
};
use num_traits::{Float, FromPrimitive, Signed};

use crate::aabb::{Bounded, IntersectsAabb};
use crate::bvh::BvhNodeBuildArgs;
#[cfg(feature = "rayon")]
use crate::bvh::rayon_executor;
use crate::point_query::PointDistance;

/// Encapsulates the required traits for the value type used in the Bvh.
pub trait BHValue:
    Scalar
    + Copy
    + FromPrimitive
    + ClosedSubAssign
    + ClosedAddAssign
    + SimdPartialOrd
    + ClosedMulAssign
    + ClosedDivAssign
    + Float
    + Signed
    + fmt::Display
{
}

impl<T> BHValue for T where
    T: Scalar
        + Copy
        + FromPrimitive
        + ClosedSubAssign
        + ClosedAddAssign
        + SimdPartialOrd
        + ClosedMulAssign
        + ClosedDivAssign
        + Float
        + Signed
        + fmt::Display
{
}

/// Describes a shape as referenced by a [`BoundingHierarchy`] leaf node.
/// Knows the index of the node in the [`BoundingHierarchy`] it is in.
///
/// [`BoundingHierarchy`]: crate::bounding_hierarchy::BoundingHierarchy
///
pub trait BHShape<T: BHValue, const D: usize>: Bounded<T, D> {
    /// Sets the index of the referenced [`BoundingHierarchy`] node.
    ///
    /// [`BoundingHierarchy`]: crate::bounding_hierarchy::BoundingHierarchy
    ///
    fn set_bh_node_index(&mut self, _: usize);

    /// Gets the index of the referenced [`BoundingHierarchy`] node.
    ///
    /// [`BoundingHierarchy`]: crate::bounding_hierarchy::BoundingHierarchy
    ///
    fn bh_node_index(&self) -> usize;
}

impl<T: BHValue, const D: usize, S: BHShape<T, D>> BHShape<T, D> for &mut S {
    fn set_bh_node_index(&mut self, idx: usize) {
        S::set_bh_node_index(self, idx)
    }

    fn bh_node_index(&self) -> usize {
        S::bh_node_index(self)
    }
}

impl<T: BHValue, const D: usize, S: BHShape<T, D>> BHShape<T, D> for Box<S> {
    fn set_bh_node_index(&mut self, idx: usize) {
        S::set_bh_node_index(self, idx)
    }

    fn bh_node_index(&self) -> usize {
        S::bh_node_index(self)
    }
}

/// This trait defines an acceleration structure with space partitioning.
/// This structure is used to efficiently compute ray-scene intersections.
pub trait BoundingHierarchy<T: BHValue, const D: usize> {
    /// Creates a new [`BoundingHierarchy`] from the `shapes` slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use bvh::bounding_hierarchy::BoundingHierarchy;
    ///
    /// # use bvh_testutil::generate_aligned_boxes;
    /// let mut shapes = generate_aligned_boxes();
    ///
    /// // Construct a normal `Bvh`.
    /// {
    ///     use bvh::bvh::Bvh;
    ///     let bvh = Bvh::build(&mut shapes);
    /// }
    ///
    /// // Or construct a `FlatBvh`.
    /// {
    ///     use bvh::flat_bvh::FlatBvh;
    ///     let bvh = FlatBvh::build(&mut shapes);
    /// }
    /// ```
    ///
    /// [`BoundingHierarchy`]: crate::bounding_hierarchy::BoundingHierarchy
    ///
    fn build<Shape: BHShape<T, D>>(shapes: &mut [Shape]) -> Self;

    /// Builds the bvh with a custom executor.
    fn build_with_executor<
        Shape: BHShape<T, D>,
        Executor: FnMut(BvhNodeBuildArgs<'_, Shape, T, D>, BvhNodeBuildArgs<'_, Shape, T, D>),
    >(
        shapes: &mut [Shape],
        executor: Executor,
    ) -> Self;

    /// Builds the bvh with a rayon based executor.
    #[cfg(feature = "rayon")]
    fn build_par<Shape: BHShape<T, D> + Send>(shapes: &mut [Shape]) -> Self
    where
        T: Send,
        Self: Sized,
    {
        Self::build_with_executor(shapes, rayon_executor)
    }

    /// Traverses the [`BoundingHierarchy`].
    /// Returns a subset of `shapes`, in which the [`Aabb`]s of the elements were hit by `ray`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bvh::bounding_hierarchy::BoundingHierarchy;
    /// use bvh::bvh::Bvh;
    /// use bvh::ray::Ray;
    /// use nalgebra::{Point3, Vector3};
    ///
    /// # use bvh_testutil::build_some_bh;
    /// let (shapes, bvh) = build_some_bh::<Bvh<f32, 3>>();
    ///
    /// // This ray traverses the x-axis and hits all 21 boxes.
    /// let origin = Point3::new(-1000.0, 0.0, 0.0);
    /// let direction = Vector3::new(1.0, 0.0, 0.0);
    /// let ray = Ray::new(origin, direction);
    /// let hit_shapes = bvh.traverse(&ray, &shapes);
    /// assert_eq!(hit_shapes.len(), 21);
    /// ```
    ///
    /// [`BoundingHierarchy`]: crate::bounding_hierarchy::BoundingHierarchy
    /// [`Aabb`]: crate::aabb::Aabb
    ///
    fn traverse<'a, Query: IntersectsAabb<T, D>, Shape: BHShape<T, D>>(
        &'a self,
        query: &Query,
        shapes: &'a [Shape],
    ) -> Vec<&'a Shape>;

    /// Traverses the [`BoundingHierarchy`].
    /// Returns the shape closest to the query point.
    ///
    /// # Examples
    ///
    /// ```
    /// use bvh::bounding_hierarchy::BoundingHierarchy;
    /// use bvh::bvh::Bvh;
    /// use nalgebra::Point3;
    ///
    /// # use bvh_testutil::build_some_bh;
    /// let (shapes, bvh) = build_some_bh::<Bvh<f32, 3>>();
    ///
    /// // The box nearest to this point is the one centered at (5, 0, 0).
    /// let query = Point3::new(5.0, 100.0, 0.0);
    /// let nearest_shape = bvh.nearest_to(query, &shapes);
    /// assert_eq!(nearest_shape.unwrap().0.id, 5);
    /// ```
    ///
    /// [`BoundingHierarchy`]: crate::bounding_hierarchy::BoundingHierarchy
    /// [`Aabb`]: crate::aabb::Aabb
    ///
    fn nearest_to<'a, Shape: BHShape<T, D> + PointDistance<T, D>>(
        &'a self,
        query: Point<T, D>,
        shapes: &'a [Shape],
    ) -> Option<(&'a Shape, T)>;

    /// Prints the [`BoundingHierarchy`] in a tree-like visualization.
    ///
    /// [`BoundingHierarchy`]: crate::bounding_hierarchy::BoundingHierarchy
    ///
    fn pretty_print(&self) {}
}

impl<T: BHValue, const D: usize, H: BoundingHierarchy<T, D>> BoundingHierarchy<T, D> for Box<H> {
    fn build<Shape: BHShape<T, D>>(shapes: &mut [Shape]) -> Self {
        Box::new(H::build(shapes))
    }

    fn traverse<'a, Query: IntersectsAabb<T, D>, Shape: BHShape<T, D>>(
        &'a self,
        query: &Query,
        shapes: &'a [Shape],
    ) -> Vec<&'a Shape> {
        H::traverse(self, query, shapes)
    }

    fn nearest_to<'a, Shape: BHShape<T, D> + PointDistance<T, D>>(
        &'a self,
        query: Point<T, D>,
        shapes: &'a [Shape],
    ) -> Option<(&'a Shape, T)> {
        H::nearest_to(self, query, shapes)
    }

    fn build_with_executor<
        Shape: BHShape<T, D>,
        Executor: FnMut(BvhNodeBuildArgs<'_, Shape, T, D>, BvhNodeBuildArgs<'_, Shape, T, D>),
    >(
        shapes: &mut [Shape],
        executor: Executor,
    ) -> Self {
        Box::new(H::build_with_executor(shapes, executor))
    }
}
