//! This module defines the [`BoundingHierarchy`] trait.

use nalgebra::{
    ClosedAddAssign, ClosedDivAssign, ClosedMulAssign, ClosedSubAssign, Point, Scalar,
    SimdPartialOrd,
};
use num::{Float, FromPrimitive, Signed};

use crate::aabb::Bounded;
#[cfg(feature = "rayon")]
use crate::bvh::rayon_executor;
use crate::bvh::BvhNodeBuildArgs;
use crate::ray::Ray;

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
    + std::fmt::Display
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
        + std::fmt::Display
{
}

/// Describes a shape as referenced by a [`BoundingHierarchy`] leaf node.
/// Knows the index of the node in the [`BoundingHierarchy`] it is in.
///
/// [`BoundingHierarchy`]: struct.BoundingHierarchy.html
///
pub trait BHShape<T: BHValue, const D: usize>: Bounded<T, D> {
    /// Sets the index of the referenced [`BoundingHierarchy`] node.
    ///
    /// [`BoundingHierarchy`]: struct.BoundingHierarchy.html
    ///
    fn set_bh_node_index(&mut self, _: usize);

    /// Gets the index of the referenced [`BoundingHierarchy`] node.
    ///
    /// [`BoundingHierarchy`]: struct.BoundingHierarchy.html
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
    /// use bvh::aabb::{Aabb, Bounded};
    /// use bvh::bounding_hierarchy::BoundingHierarchy;
    /// use nalgebra::{Point3, Vector3};
    /// # use bvh::bounding_hierarchy::BHShape;
    /// # pub struct UnitBox {
    /// #     pub id: i32,
    /// #     pub pos: Point3<f32>,
    /// #     node_index: usize,
    /// # }
    /// #
    /// # impl UnitBox {
    /// #     pub fn new(id: i32, pos: Point3<f32>) -> UnitBox {
    /// #         UnitBox {
    /// #             id: id,
    /// #             pos: pos,
    /// #             node_index: 0,
    /// #         }
    /// #     }
    /// # }
    /// #
    /// # impl Bounded<f32,3> for UnitBox {
    /// #     fn aabb(&self) -> Aabb<f32,3> {
    /// #         let min = self.pos + Vector3::new(-0.5, -0.5, -0.5);
    /// #         let max = self.pos + Vector3::new(0.5, 0.5, 0.5);
    /// #         Aabb::with_bounds(min, max)
    /// #     }
    /// # }
    /// #
    /// # impl BHShape<f32,3> for UnitBox {
    /// #     fn set_bh_node_index(&mut self, index: usize) {
    /// #         self.node_index = index;
    /// #     }
    /// #
    /// #     fn bh_node_index(&self) -> usize {
    /// #         self.node_index
    /// #     }
    /// # }
    /// #
    /// # fn create_bhshapes() -> Vec<UnitBox> {
    /// #     let mut shapes = Vec::new();
    /// #     for i in 0..1000 {
    /// #         let position = Point3::new(i as f32, i as f32, i as f32);
    /// #         shapes.push(UnitBox::new(i, position));
    /// #     }
    /// #     shapes
    /// # }
    ///
    /// let mut shapes = create_bhshapes();
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
    /// [`BoundingHierarchy`]: trait.BoundingHierarchy.html
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
    /// use bvh::aabb::{Aabb, Bounded};
    /// use bvh::bounding_hierarchy::BoundingHierarchy;
    /// use bvh::bvh::Bvh;
    /// use nalgebra::{Point3, Vector3};
    /// use bvh::ray::Ray;
    /// # use bvh::bounding_hierarchy::BHShape;
    /// # pub struct UnitBox {
    /// #     pub id: i32,
    /// #     pub pos: Point3<f32>,
    /// #     node_index: usize,
    /// # }
    /// #
    /// # impl UnitBox {
    /// #     pub fn new(id: i32, pos: Point3<f32>) -> UnitBox {
    /// #         UnitBox {
    /// #             id: id,
    /// #             pos: pos,
    /// #             node_index: 0,
    /// #         }
    /// #     }
    /// # }
    /// #
    /// # impl Bounded<f32,3> for UnitBox {
    /// #     fn aabb(&self) -> Aabb<f32,3> {
    /// #         let min = self.pos + Vector3::new(-0.5, -0.5, -0.5);
    /// #         let max = self.pos + Vector3::new(0.5, 0.5, 0.5);
    /// #         Aabb::with_bounds(min, max)
    /// #     }
    /// # }
    /// #
    /// # impl BHShape<f32,3> for UnitBox {
    /// #     fn set_bh_node_index(&mut self, index: usize) {
    /// #         self.node_index = index;
    /// #     }
    /// #
    /// #     fn bh_node_index(&self) -> usize {
    /// #         self.node_index
    /// #     }
    /// # }
    /// #
    /// # fn create_bvh() -> (Bvh<f32,3>, Vec<UnitBox>) {
    /// #     let mut shapes = Vec::new();
    /// #     for i in 0..1000 {
    /// #         let position = Point3::new(i as f32, i as f32, i as f32);
    /// #         shapes.push(UnitBox::new(i, position));
    /// #     }
    /// #     let bvh = Bvh::build(&mut shapes);
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
    /// [`Aabb`]: ../aabb/struct.Aabb.html
    ///
    fn traverse<'a, Shape: BHShape<T, D>>(
        &'a self,
        ray: &Ray<T, D>,
        shapes: &'a [Shape],
    ) -> Vec<&'a Shape>;

    /// Traverses the [`BoundingHierarchy`].
    /// Returns a subset of `shapes` which are candidates for being the closest to the query point.
    ///
    /// # Examples
    ///
    /// ```
    /// use bvh::aabb::{Aabb, Bounded};
    /// use bvh::bounding_hierarchy::BoundingHierarchy;
    /// use bvh::bvh::Bvh;
    /// use nalgebra::{Point3, Vector3};
    /// use bvh::ray::Ray;
    /// # use bvh::bounding_hierarchy::BHShape;
    /// # pub struct UnitBox {
    /// #     pub id: i32,
    /// #     pub pos: Point3<f32>,
    /// #     node_index: usize,
    /// # }
    /// #
    /// # impl UnitBox {
    /// #     pub fn new(id: i32, pos: Point3<f32>) -> UnitBox {
    /// #         UnitBox {
    /// #             id: id,
    /// #             pos: pos,
    /// #             node_index: 0,
    /// #         }
    /// #     }
    /// # }
    /// #
    /// # impl Bounded<f32,3> for UnitBox {
    /// #     fn aabb(&self) -> Aabb<f32,3> {
    /// #         let min = self.pos + Vector3::new(-0.5, -0.5, -0.5);
    /// #         let max = self.pos + Vector3::new(0.5, 0.5, 0.5);
    /// #         Aabb::with_bounds(min, max)
    /// #     }
    /// # }
    /// #
    /// # impl BHShape<f32,3> for UnitBox {
    /// #     fn set_bh_node_index(&mut self, index: usize) {
    /// #         self.node_index = index;
    /// #     }
    /// #
    /// #     fn bh_node_index(&self) -> usize {
    /// #         self.node_index
    /// #     }
    /// # }
    /// #
    /// # fn create_bvh() -> (Bvh<f32,3>, Vec<UnitBox>) {
    /// #     let mut shapes = Vec::new();
    /// #     for i in 0..1000 {
    /// #         let position = Point3::new(i as f32, i as f32, i as f32);
    /// #         shapes.push(UnitBox::new(i, position));
    /// #     }
    /// #     let bvh = Bvh::build(&mut shapes);
    /// #     (bvh, shapes)
    /// # }
    ///
    /// let (bvh, shapes) = create_bvh();
    ///
    /// let query = Point3::new(5.0, 0.0, 2.0);
    /// let direction = Vector3::new(1.0, 0.0, 0.0);
    /// let hit_shapes = bvh.nearest_candidates(&query, &shapes);
    /// ```
    ///
    /// [`BoundingHierarchy`]: trait.BoundingHierarchy.html
    /// [`Aabb`]: ../aabb/struct.Aabb.html
    ///
    fn nearest_candidates<'a, Shape: BHShape<T, D>>(
        &'a self,
        query: &Point<T, D>,
        shapes: &'a [Shape],
    ) -> Vec<&Shape>;

    /// Prints the [`BoundingHierarchy`] in a tree-like visualization.
    ///
    /// [`BoundingHierarchy`]: trait.BoundingHierarchy.html
    ///
    fn pretty_print(&self) {}
}

impl<T: BHValue, const D: usize, H: BoundingHierarchy<T, D>> BoundingHierarchy<T, D> for Box<H> {
    fn build<Shape: BHShape<T, D>>(shapes: &mut [Shape]) -> Self {
        Box::new(H::build(shapes))
    }

    fn traverse<'a, Shape: BHShape<T, D>>(
        &'a self,
        ray: &Ray<T, D>,
        shapes: &'a [Shape],
    ) -> Vec<&'a Shape> {
        H::traverse(self, ray, shapes)
    }

    fn nearest_candidates<'a, Shape: BHShape<T, D>>(
        &'a self,
        query: &Point<T, D>,
        shapes: &'a [Shape],
    ) -> Vec<&Shape> {
        H::nearest_candidates(self, query, shapes)
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
