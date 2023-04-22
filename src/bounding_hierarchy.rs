//! This module defines the `BoundingHierarchy` trait.

use nalgebra::Scalar;

use crate::aabb::Bounded;
use crate::ray::Ray;

/// Describes a shape as referenced by a [`BoundingHierarchy`] leaf node.
/// Knows the index of the node in the [`BoundingHierarchy`] it is in.
///
/// [`BoundingHierarchy`]: struct.BoundingHierarchy.html
///
#[allow(clippy::upper_case_acronyms)]
pub trait BHShape<T: Scalar + Copy, const D: usize>: Bounded<T, D> {
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

impl<T: Scalar + Copy, const D: usize, S: BHShape<T, D>> BHShape<T, D> for &mut S {
    fn set_bh_node_index(&mut self, idx: usize) {
        S::set_bh_node_index(self, idx)
    }

    fn bh_node_index(&self) -> usize {
        S::bh_node_index(self)
    }
}

impl<T: Scalar + Copy, const D: usize, S: BHShape<T, D>> BHShape<T, D> for Box<S> {
    fn set_bh_node_index(&mut self, idx: usize) {
        S::set_bh_node_index(self, idx)
    }

    fn bh_node_index(&self) -> usize {
        S::bh_node_index(self)
    }
}

/// This trait defines an acceleration structure with space partitioning.
/// This structure is used to efficiently compute ray-scene intersections.
pub trait BoundingHierarchy<T: Scalar + Copy, const D: usize> {
    /// Creates a new [`BoundingHierarchy`] from the `shapes` slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use bvh::aabb::{AABB, Bounded};
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
    /// #     fn aabb(&self) -> AABB<f32,3> {
    /// #         let min = self.pos + Vector3::new(-0.5, -0.5, -0.5);
    /// #         let max = self.pos + Vector3::new(0.5, 0.5, 0.5);
    /// #         AABB::with_bounds(min, max)
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
    /// // Construct a normal `BVH`.
    /// {
    ///     use bvh::bvh::BVH;
    ///     let bvh = BVH::build(&mut shapes);
    /// }
    ///
    /// // Or construct a `FlatBVH`.
    /// {
    ///     use bvh::flat_bvh::FlatBVH;
    ///     let bvh = FlatBVH::build(&mut shapes);
    /// }
    /// ```
    ///
    /// [`BoundingHierarchy`]: trait.BoundingHierarchy.html
    ///
    fn build<Shape: BHShape<T, D>>(shapes: &mut [Shape]) -> Self;

    /// Traverses the [`BoundingHierarchy`].
    /// Returns a subset of `shapes`, in which the [`AABB`]s of the elements were hit by `ray`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bvh::aabb::{AABB, Bounded};
    /// use bvh::bounding_hierarchy::BoundingHierarchy;
    /// use bvh::bvh::BVH;
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
    /// #     fn aabb(&self) -> AABB<f32,3> {
    /// #         let min = self.pos + Vector3::new(-0.5, -0.5, -0.5);
    /// #         let max = self.pos + Vector3::new(0.5, 0.5, 0.5);
    /// #         AABB::with_bounds(min, max)
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
    /// # fn create_bvh() -> (BVH<f32,3>, Vec<UnitBox>) {
    /// #     let mut shapes = Vec::new();
    /// #     for i in 0..1000 {
    /// #         let position = Point3::new(i as f32, i as f32, i as f32);
    /// #         shapes.push(UnitBox::new(i, position));
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
    fn traverse<'a, Shape: BHShape<T, D>>(
        &'a self,
        ray: &Ray<T, D>,
        shapes: &'a [Shape],
    ) -> Vec<&Shape>;

    /// Prints the [`BoundingHierarchy`] in a tree-like visualization.
    ///
    /// [`BoundingHierarchy`]: trait.BoundingHierarchy.html
    ///
    fn pretty_print(&self) {}
}

impl<T: Scalar + Copy, const D: usize, H: BoundingHierarchy<T, D>> BoundingHierarchy<T, D>
    for Box<H>
{
    fn build<Shape: BHShape<T, D>>(shapes: &mut [Shape]) -> Self {
        Box::new(H::build(shapes))
    }

    fn traverse<'a, Shape: BHShape<T, D>>(
        &'a self,
        ray: &Ray<T, D>,
        shapes: &'a [Shape],
    ) -> Vec<&Shape> {
        H::traverse(self, ray, shapes)
    }
}
