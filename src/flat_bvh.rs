//! This module exports methods to flatten the [`Bvh`] into a [`FlatBvh`] and traverse it iteratively.
use crate::aabb::{Aabb, Bounded, IntersectsAabb};
use crate::bounding_hierarchy::{BHShape, BHValue, BoundingHierarchy};
use crate::bvh::{Bvh, BvhNode};
use crate::point_query::PointDistance;

use nalgebra::Point;
use num_traits::Float;

/// A structure of a node of a flat [`Bvh`]. The structure of the nodes allows for an
/// iterative traversal approach without the necessity to maintain a stack or queue.
///
/// [`Bvh`]: ../bvh/struct.Bvh.html
///
pub struct FlatNode<T: BHValue, const D: usize> {
    /// The [`Aabb`] of the [`Bvh`] node. Prior to testing the [`Aabb`] bounds,
    /// the `entry_index` must be checked. In case the entry_index is [`u32::MAX`],
    /// the [`Aabb`] is undefined.
    ///
    /// [`Aabb`]: ../aabb/struct.Aabb.html
    /// [`Bvh`]: ../bvh/struct.Bvh.html
    /// [`u32::MAX`]: https://doc.rust-lang.org/std/u32/constant.MAX.html
    ///
    pub aabb: Aabb<T, D>,

    /// The index of the `FlatNode` to jump to, if the [`Aabb`] test is positive.
    /// If this value is [`u32::MAX`] then the current node is a leaf node.
    /// Leaf nodes contain a shape index and an exit index. In leaf nodes the
    /// [`Aabb`] is undefined.
    ///
    /// [`Aabb`]: ../aabb/struct.Aabb.html
    /// [`u32::MAX`]: https://doc.rust-lang.org/std/u32/constant.MAX.html
    ///
    pub entry_index: u32,

    /// The index of the [`FlatNode`] to jump to, if the [`Aabb`] test is negative.
    ///
    /// [`Aabb`]: ../aabb/struct.Aabb.html
    ///
    pub exit_index: u32,

    /// The index of the shape in the shapes array.
    pub shape_index: u32,
}

impl<T: BHValue, const D: usize> FlatNode<T, D> {
    /// Returns whether the node is a leaf node.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.entry_index == u32::MAX
    }
}

impl<T: BHValue + Float, const D: usize> BvhNode<T, D> {
    /// Creates a flat node from a `Bvh` inner node and its [`Aabb`]. Returns the next free index.
    /// TODO: change the algorithm which pushes `FlatNode`s to a vector to not use indices this
    /// much. Implement an algorithm which writes directly to a writable slice.
    fn create_flat_branch<F, FNodeType>(
        &self,
        nodes: &[BvhNode<T, D>],
        this_aabb: &Aabb<T, D>,
        vec: &mut Vec<FNodeType>,
        next_free: usize,
        constructor: &F,
    ) -> usize
    where
        F: Fn(&Aabb<T, D>, u32, u32, u32) -> FNodeType,
    {
        // Create dummy node.
        let dummy = constructor(&Aabb::empty(), 0, 0, 0);
        vec.push(dummy);
        assert_eq!(vec.len() - 1, next_free);

        // Create subtree.
        let index_after_subtree = self.flatten_custom(nodes, vec, next_free + 1, constructor);

        // Replace dummy node by actual node with the entry index pointing to the subtree
        // and the exit index pointing to the next node after the subtree.
        let navigator_node = constructor(
            this_aabb,
            (next_free + 1) as u32,
            index_after_subtree as u32,
            u32::MAX,
        );
        vec[next_free] = navigator_node;
        index_after_subtree
    }

    /// Flattens the [`Bvh`], so that it can be traversed in an iterative manner.
    /// This method constructs custom flat nodes using the `constructor`.
    ///
    /// [`Bvh`]: ../bvh/struct.Bvh.html
    ///
    pub fn flatten_custom<F, FNodeType>(
        &self,
        nodes: &[BvhNode<T, D>],
        vec: &mut Vec<FNodeType>,
        next_free: usize,
        constructor: &F,
    ) -> usize
    where
        F: Fn(&Aabb<T, D>, u32, u32, u32) -> FNodeType,
    {
        match *self {
            BvhNode::Node {
                ref child_l_aabb,
                child_l_index,
                ref child_r_aabb,
                child_r_index,
                ..
            } => {
                let index_after_child_l = nodes[child_l_index].create_flat_branch(
                    nodes,
                    child_l_aabb,
                    vec,
                    next_free,
                    constructor,
                );
                nodes[child_r_index].create_flat_branch(
                    nodes,
                    child_r_aabb,
                    vec,
                    index_after_child_l,
                    constructor,
                )
            }
            BvhNode::Leaf { shape_index, .. } => {
                let mut next_shape = next_free;
                next_shape += 1;
                let leaf_node = constructor(
                    &Aabb::empty(),
                    u32::MAX,
                    next_shape as u32,
                    shape_index as u32,
                );
                vec.push(leaf_node);

                next_shape
            }
        }
    }
}

/// A flat [`Bvh`]. Represented by a vector of [`FlatNode`]s. The [`FlatBvh`] is designed for use
/// where a recursive traversal of a data structure is not possible, for example shader programs.
///
/// [`Bvh`]: ../bvh/struct.Bvh.html
/// [`FlatNode`]: struct.FlatNode.html
/// [`FlatBvh`]: struct.FlatBvh.html
///
pub type FlatBvh<T, const D: usize> = Vec<FlatNode<T, D>>;

impl<T: BHValue, const D: usize> Bvh<T, D> {
    /// Flattens the [`Bvh`] so that it can be traversed iteratively.
    /// Constructs the flat nodes using the supplied function.
    /// This function can be used, when the flat bvh nodes should be of some particular
    /// non-default structure.
    /// The `constructor` is fed the following arguments in this order:
    ///
    /// 1 - &Aabb: The enclosing [`Aabb`]
    /// 2 - u32: The index of the nested node
    /// 3 - u32: The exit index
    /// 4 - u32: The shape index
    ///
    /// [`Bvh`]: ../bvh/struct.Bvh.html
    ///
    /// # Example
    ///
    /// ```
    /// use bvh::aabb::{Aabb, Bounded};
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
    /// # fn create_bhshapes() -> Vec<UnitBox> {
    /// #     let mut shapes = Vec::new();
    /// #     for i in 0..1000 {
    /// #         let position = Point3::new(i as f32, i as f32, i as f32);
    /// #         shapes.push(UnitBox::new(i, position));
    /// #     }
    /// #     shapes
    /// # }
    ///
    /// struct CustomStruct {
    ///     aabb: Aabb<f32,3>,
    ///     entry_index: u32,
    ///     exit_index: u32,
    ///     shape_index: u32,
    /// }
    ///
    /// let custom_constructor = |aabb: &Aabb<f32,3>, entry, exit, shape_index| {
    ///     CustomStruct {
    ///         aabb: *aabb,
    ///         entry_index: entry,
    ///         exit_index: exit,
    ///         shape_index: shape_index,
    ///     }
    /// };
    ///
    /// let mut shapes = create_bhshapes();
    /// let bvh = Bvh::build(&mut shapes);
    /// let custom_flat_bvh = bvh.flatten_custom(&custom_constructor);
    /// ```
    pub fn flatten_custom<F, FNodeType>(&self, constructor: &F) -> Vec<FNodeType>
    where
        F: Fn(&Aabb<T, D>, u32, u32, u32) -> FNodeType,
    {
        let mut vec = Vec::new();
        if self.nodes.is_empty() {
            // There is no node_index=0.
            return vec;
        }
        self.nodes[0].flatten_custom(&self.nodes, &mut vec, 0, constructor);
        vec
    }

    /// Flattens the [`Bvh`] so that it can be traversed iteratively.
    ///
    /// [`Bvh`]: ../bvh/struct.Bvh.html
    ///
    /// # Example
    ///
    /// ```
    /// use bvh::aabb::{Aabb, Bounded};
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
    /// let bvh = Bvh::build(&mut shapes);
    /// let flat_bvh = bvh.flatten();
    /// ```
    pub fn flatten(&self) -> FlatBvh<T, D> {
        self.flatten_custom(&|aabb, entry, exit, shape| FlatNode {
            aabb: *aabb,
            entry_index: entry,
            exit_index: exit,
            shape_index: shape,
        })
    }
}

impl<T: BHValue + std::fmt::Display, const D: usize> BoundingHierarchy<T, D> for FlatBvh<T, D> {
    /// A [`FlatBvh`] is built from a regular [`Bvh`] using the [`Bvh::flatten`] method.
    ///
    /// [`FlatBvh`]: struct.FlatBvh.html
    /// [`Bvh`]: ../bvh/struct.Bvh.html
    ///
    fn build<Shape: BHShape<T, D>>(shapes: &mut [Shape]) -> FlatBvh<T, D> {
        let bvh = Bvh::build(shapes);
        bvh.flatten()
    }

    /// Traverses a [`FlatBvh`] structure iteratively.
    ///
    /// [`FlatBvh`]: struct.FlatBvh.html
    ///
    /// # Examples
    ///
    /// ```
    /// use bvh::aabb::{Aabb, Bounded};
    /// use bvh::bounding_hierarchy::BoundingHierarchy;
    /// use bvh::flat_bvh::FlatBvh;
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
    /// # fn create_bhshapes() -> Vec<UnitBox> {
    /// #     let mut shapes = Vec::new();
    /// #     for i in 0..1000 {
    /// #         let position = Point3::new(i as f32, i as f32, i as f32);
    /// #         shapes.push(UnitBox::new(i, position));
    /// #     }
    /// #     shapes
    /// # }
    ///
    /// let origin = Point3::new(0.0,0.0,0.0);
    /// let direction = Vector3::new(1.0,0.0,0.0);
    /// let ray = Ray::new(origin, direction);
    /// let mut shapes = create_bhshapes();
    /// let flat_bvh = FlatBvh::build(&mut shapes);
    /// let hit_shapes = flat_bvh.traverse(&ray, &shapes);
    /// ```
    fn traverse<'a, Q: IntersectsAabb<T, D>, B: Bounded<T, D>>(
        &'a self,
        query: &Q,
        shapes: &'a [B],
    ) -> Vec<&'a B> {
        let mut hit_shapes = Vec::new();
        let mut index = 0;

        // The traversal loop should terminate when `max_length` is set as the next node index.
        let max_length = self.len();

        // Iterate while the node index is valid.
        while index < max_length {
            let node = &self[index];

            if node.is_leaf() {
                let shape = &shapes[node.shape_index as usize];
                if query.intersects_aabb(&shape.aabb()) {
                    hit_shapes.push(shape);
                }

                // Exit the current node.
                index = node.exit_index as usize;
            } else if query.intersects_aabb(&node.aabb) {
                // If entry_index is not MAX_UINT32 and the Aabb test passes, then
                // proceed to the node in entry_index (which goes down the bvh branch).
                index = node.entry_index as usize;
            } else {
                // If entry_index is not MAX_UINT32 and the Aabb test fails, then
                // proceed to the node in exit_index (which defines the next untested partition).
                index = node.exit_index as usize;
            }
        }

        hit_shapes
    }

    /// Traverses a [`FlatBvh`] structure iteratively.
    /// Returns the nearest shape to the query point and the distance to it.
    ///
    /// Note: the flat bvh is not optimized for nearest queries. It is recommended to use
    /// a [`Bvh`] for nearest queries when possible.
    /// The flat bvh will still be much faster than a linear search.
    ///
    /// [`FlatBvh`]: struct.FlatBvh.html
    ///
    /// # Examples
    ///
    /// ```
    /// use bvh::aabb::{Aabb, Bounded};
    /// use bvh::bounding_hierarchy::BoundingHierarchy;
    /// use bvh::flat_bvh::FlatBvh;
    /// use bvh::point_query::PointDistance;
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
    /// # impl PointDistance<f32,3> for UnitBox {
    /// #     fn distance_squared(&self, query_point: Point3<f32>) -> f32 {
    /// #         self.aabb().min_distance_squared(query_point)
    /// #    }
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
    /// let flat_bvh = FlatBvh::build(&mut shapes);
    ///
    /// let query = Point3::new(5.0, 5.7, 5.3);
    /// let nearest_shape = flat_bvh.nearest_to(query, &shapes);
    ///
    /// # assert_eq!(nearest_shape.unwrap().0.id, 5);
    /// ```
    ///
    /// [`BoundingHierarchy`]: trait.BoundingHierarchy.html
    /// [`Aabb`]: ../aabb/struct.Aabb.html
    ///
    fn nearest_to<'a, Shape: BHShape<T, D> + PointDistance<T, D>>(
        &'a self,
        query: Point<T, D>,
        shapes: &'a [Shape],
    ) -> Option<(&'a Shape, T)> {
        if self.is_empty() {
            return None;
        }

        let mut best_element = None;

        let mut index = 0;

        // The traversal loop should terminate when `max_length` is set as the next node index.
        let max_length = self.len();

        // Iterate while the node index is valid.
        while index < max_length {
            let node = &self[index];

            if node.is_leaf() {
                let shape = &shapes[node.shape_index as usize];
                let dist = shape.distance_squared(query);

                // TODO: to be replaced by `Option::is_none_or` after 2025-10 for 1 year MSRV.
                #[allow(clippy::unnecessary_map_or)]
                if best_element.map_or(true, |(_, best_dist)| dist < best_dist) {
                    best_element = Some((shape, dist));
                }

                // Exit the current node.
                index = node.exit_index as usize;
            } else {
                let min_dist = node.aabb.min_distance_squared(query);

                // TODO: to be replaced by `Option::is_none_or` after 2025-10 for 1 year MSRV.
                #[allow(clippy::unnecessary_map_or)]
                if best_element.map_or(true, |(_, best_dist)| min_dist < best_dist) {
                    // This node needs further traversal.
                    index = node.entry_index as usize;
                } else {
                    // This node can't contain a candidate shape.
                    index = node.exit_index as usize;
                }
            }
        }

        // Return the best shape and its distance. We had a distance squared previously.
        best_element.map(|best| (best.0, best.1.sqrt()))
    }

    /// Prints a textual representation of a [`FlatBvh`].
    ///
    /// [`FlatBvh`]: struct.FlatBvh.html
    ///
    fn pretty_print(&self) {
        for (i, node) in self.iter().enumerate() {
            println!(
                "{}\tentry {}\texit {}\tshape {}",
                i, node.entry_index, node.exit_index, node.shape_index
            );
        }
    }

    fn build_with_executor<
        Shape: BHShape<T, D>,
        Executor: FnMut(
            crate::bvh::BvhNodeBuildArgs<'_, Shape, T, D>,
            crate::bvh::BvhNodeBuildArgs<'_, Shape, T, D>,
        ),
    >(
        shapes: &mut [Shape],
        executor: Executor,
    ) -> Self {
        let bvh = Bvh::build_with_executor(shapes, executor);
        bvh.flatten()
    }
}

#[cfg(test)]
mod tests {
    use crate::testbase::{
        build_empty_bh, build_some_bh, nearest_to_some_bh, traverse_some_bh, TBvh3, TFlatBvh3,
    };

    #[test]
    /// Tests whether the building procedure succeeds in not failing.
    fn test_build_flat_bvh() {
        build_some_bh::<TFlatBvh3>();
    }

    #[test]
    /// Runs some primitive tests for intersections of a ray with a fixed scene given
    /// as a `FlatBvh`.
    fn test_traverse_flat_bvh() {
        traverse_some_bh::<TFlatBvh3>();
    }

    #[test]
    /// Runs some primitive tests for distance query of a point with a fixed scene given as a [`Bvh`].
    fn test_nearest_to_flat_bvh() {
        nearest_to_some_bh::<TFlatBvh3>();
    }
    #[test]
    fn test_flatten_empty_bvh() {
        let (_, bvh) = build_empty_bh::<TBvh3>();
        let flat = bvh.flatten();
        assert!(flat.is_empty());
    }
}

#[cfg(all(feature = "bench", test))]
mod bench {
    use crate::testbase::{
        build_1200_triangles_bh, build_120k_triangles_bh, build_12k_triangles_bh, create_n_cubes,
        default_bounds, intersect_1200_triangles_bh, intersect_120k_triangles_bh,
        intersect_12k_triangles_bh, nearest_to_1200_triangles_bh, nearest_to_120k_triangles_bh,
        nearest_to_12k_triangles_bh, TBvh3, TFlatBvh3,
    };

    #[bench]
    /// Benchmark the flattening of a [`Bvh`] with 120,000 triangles.
    fn bench_flatten_120k_triangles_bvh(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        let bvh = TBvh3::build(&mut triangles);

        b.iter(|| {
            bvh.flatten();
        });
    }
    #[bench]
    /// Benchmark the construction of a [`FlatBvh`] with 1,200 triangles.
    fn bench_build_1200_triangles_flat_bvh(b: &mut ::test::Bencher) {
        build_1200_triangles_bh::<TFlatBvh3>(b);
    }

    #[bench]
    /// Benchmark the construction of a [`FlatBvh`] with 12,000 triangles.
    fn bench_build_12k_triangles_flat_bvh(b: &mut ::test::Bencher) {
        build_12k_triangles_bh::<TFlatBvh3>(b);
    }

    #[bench]
    /// Benchmark the construction of a [`FlatBvh`] with 120,000 triangles.
    fn bench_build_120k_triangles_flat_bvh(b: &mut ::test::Bencher) {
        build_120k_triangles_bh::<TFlatBvh3>(b);
    }

    #[bench]
    /// Benchmark intersecting 1,200 triangles using the recursive [`FlatBvh`].
    fn bench_intersect_1200_triangles_flat_bvh(b: &mut ::test::Bencher) {
        intersect_1200_triangles_bh::<TFlatBvh3>(b);
    }

    #[bench]
    /// Benchmark intersecting 12,000 triangles using the recursive [`FlatBvh`].
    fn bench_intersect_12k_triangles_flat_bvh(b: &mut ::test::Bencher) {
        intersect_12k_triangles_bh::<TFlatBvh3>(b);
    }

    #[bench]
    /// Benchmark intersecting 120,000 triangles using the recursive [`FlatBvh`].
    fn bench_intersect_120k_triangles_flat_bvh(b: &mut ::test::Bencher) {
        intersect_120k_triangles_bh::<TFlatBvh3>(b);
    }

    #[bench]
    /// Benchmark nearest_to on 1,200 triangles using the recursive [`FlatBvh`].
    fn bench_nearest_to_1200_triangles_bvh(b: &mut ::test::Bencher) {
        nearest_to_1200_triangles_bh::<TFlatBvh3>(b);
    }

    #[bench]
    /// Benchmark nearest_to on 12,000 triangles using the recursive [`FlatBvh`].
    fn bench_nearest_to_12k_triangles_bvh(b: &mut ::test::Bencher) {
        nearest_to_12k_triangles_bh::<TFlatBvh3>(b);
    }

    #[bench]
    /// Benchmark nearest_to on 120,000 triangles using the recursive [`FlatBvh`].
    fn bench_nearest_to_120k_triangles_bvh(b: &mut ::test::Bencher) {
        nearest_to_120k_triangles_bh::<TFlatBvh3>(b);
    }
}
