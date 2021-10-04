//! This module exports methods to flatten the `BVH` and traverse it iteratively.

use crate::aabb::{Bounded, AABB};
use crate::bounding_hierarchy::{BHShape, BoundingHierarchy};
use crate::bvh::{BVHNode, BVH};
use crate::ray::Ray;

/// A structure of a node of a flat [`BVH`]. The structure of the nodes allows for an
/// iterative traversal approach without the necessity to maintain a stack or queue.
///
/// [`BVH`]: ../bvh/struct.BVH.html
///
pub struct FlatNode {
    /// The [`AABB`] of the [`BVH`] node. Prior to testing the [`AABB`] bounds,
    /// the `entry_index` must be checked. In case the entry_index is [`u32::max_value()`],
    /// the [`AABB`] is undefined.
    ///
    /// [`AABB`]: ../aabb/struct.AABB.html
    /// [`BVH`]: ../bvh/struct.BVH.html
    /// [`u32::max_value()`]: https://doc.rust-lang.org/std/u32/constant.MAX.html
    ///
    pub aabb: AABB,

    /// The index of the `FlatNode` to jump to, if the [`AABB`] test is positive.
    /// If this value is [`u32::max_value()`] then the current node is a leaf node.
    /// Leaf nodes contain a shape index and an exit index. In leaf nodes the
    /// [`AABB`] is undefined.
    ///
    /// [`AABB`]: ../aabb/struct.AABB.html
    /// [`u32::max_value()`]: https://doc.rust-lang.org/std/u32/constant.MAX.html
    ///
    pub entry_index: u32,

    /// The index of the `FlatNode` to jump to, if the [`AABB`] test is negative.
    ///
    /// [`AABB`]: ../aabb/struct.AABB.html
    ///
    pub exit_index: u32,

    /// The index of the shape in the shapes array.
    pub shape_index: u32,
}

impl BVHNode {
    /// Creates a flat node from a `BVH` inner node and its `AABB`. Returns the next free index.
    /// TODO: change the algorithm which pushes `FlatNode`s to a vector to not use indices this
    /// much. Implement an algorithm which writes directly to a writable slice.
    fn create_flat_branch<F, FNodeType>(
        &self,
        nodes: &[BVHNode],
        this_aabb: &AABB,
        vec: &mut Vec<FNodeType>,
        next_free: usize,
        constructor: &F,
    ) -> usize
    where
        F: Fn(&AABB, u32, u32, u32) -> FNodeType,
    {
        // Create dummy node.
        let dummy = constructor(&AABB::empty(), 0, 0, 0);
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
            u32::max_value(),
        );
        vec[next_free] = navigator_node;
        index_after_subtree
    }

    /// Flattens the [`BVH`], so that it can be traversed in an iterative manner.
    /// This method constructs custom flat nodes using the `constructor`.
    ///
    /// [`BVH`]: ../bvh/struct.BVH.html
    ///
    pub fn flatten_custom<F, FNodeType>(
        &self,
        nodes: &[BVHNode],
        vec: &mut Vec<FNodeType>,
        next_free: usize,
        constructor: &F,
    ) -> usize
    where
        F: Fn(&AABB, u32, u32, u32) -> FNodeType,
    {
        match *self {
            BVHNode::Node {
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
            BVHNode::Leaf { shape_index, .. } => {
                let mut next_shape = next_free;
                next_shape += 1;
                let leaf_node = constructor(
                    &AABB::empty(),
                    u32::max_value(),
                    next_shape as u32,
                    shape_index as u32,
                );
                vec.push(leaf_node);

                next_shape
            }
        }
    }
}

/// A flat [`BVH`]. Represented by a vector of [`FlatNode`]s. The [`FlatBVH`] is designed for use
/// where a recursive traversal of a data structure is not possible, for example shader programs.
///
/// [`BVH`]: ../bvh/struct.BVH.html
/// [`FlatNode`]: struct.FlatNode.html
/// [`FlatBVH`]: struct.FlatBVH.html
///
#[allow(clippy::upper_case_acronyms)]
pub type FlatBVH = Vec<FlatNode>;

impl BVH {
    /// Flattens the [`BVH`] so that it can be traversed iteratively.
    /// Constructs the flat nodes using the supplied function.
    /// This function can be used, when the flat bvh nodes should be of some particular
    /// non-default structure.
    /// The `constructor` is fed the following arguments in this order:
    ///
    /// 1 - &AABB: The enclosing `AABB`
    /// 2 - u32: The index of the nested node
    /// 3 - u32: The exit index
    /// 4 - u32: The shape index
    ///
    /// [`BVH`]: ../bvh/struct.BVH.html
    ///
    /// # Example
    ///
    /// ```
    /// use bvh::aabb::{AABB, Bounded};
    /// use bvh::bvh::BVH;
    /// use bvh::{Point3, Vector3};
    /// use bvh::ray::Ray;
    /// # use bvh::bounding_hierarchy::BHShape;
    /// # pub struct UnitBox {
    /// #     pub id: i32,
    /// #     pub pos: Point3,
    /// #     node_index: usize,
    /// # }
    /// #
    /// # impl UnitBox {
    /// #     pub fn new(id: i32, pos: Point3) -> UnitBox {
    /// #         UnitBox {
    /// #             id: id,
    /// #             pos: pos,
    /// #             node_index: 0,
    /// #         }
    /// #     }
    /// # }
    /// #
    /// # impl Bounded for UnitBox {
    /// #     fn aabb(&self) -> AABB {
    /// #         let min = self.pos + Vector3::new(-0.5, -0.5, -0.5);
    /// #         let max = self.pos + Vector3::new(0.5, 0.5, 0.5);
    /// #         AABB::with_bounds(min, max)
    /// #     }
    /// # }
    /// #
    /// # impl BHShape for UnitBox {
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
    ///     aabb: AABB,
    ///     entry_index: u32,
    ///     exit_index: u32,
    ///     shape_index: u32,
    /// }
    ///
    /// let custom_constructor = |aabb: &AABB, entry, exit, shape_index| {
    ///     CustomStruct {
    ///         aabb: *aabb,
    ///         entry_index: entry,
    ///         exit_index: exit,
    ///         shape_index: shape_index,
    ///     }
    /// };
    ///
    /// let mut shapes = create_bhshapes();
    /// let bvh = BVH::build(&mut shapes);
    /// let custom_flat_bvh = bvh.flatten_custom(&custom_constructor);
    /// ```
    pub fn flatten_custom<F, FNodeType>(&self, constructor: &F) -> Vec<FNodeType>
    where
        F: Fn(&AABB, u32, u32, u32) -> FNodeType,
    {
        let mut vec = Vec::new();
        self.nodes[0].flatten_custom(&self.nodes, &mut vec, 0, constructor);
        vec
    }

    /// Flattens the [`BVH`] so that it can be traversed iteratively.
    ///
    /// [`BVH`]: ../bvh/struct.BVH.html
    ///
    /// # Example
    ///
    /// ```
    /// use bvh::aabb::{AABB, Bounded};
    /// use bvh::bvh::BVH;
    /// use bvh::{Point3, Vector3};
    /// use bvh::ray::Ray;
    /// # use bvh::bounding_hierarchy::BHShape;
    /// # pub struct UnitBox {
    /// #     pub id: i32,
    /// #     pub pos: Point3,
    /// #     node_index: usize,
    /// # }
    /// #
    /// # impl UnitBox {
    /// #     pub fn new(id: i32, pos: Point3) -> UnitBox {
    /// #         UnitBox {
    /// #             id: id,
    /// #             pos: pos,
    /// #             node_index: 0,
    /// #         }
    /// #     }
    /// # }
    /// #
    /// # impl Bounded for UnitBox {
    /// #     fn aabb(&self) -> AABB {
    /// #         let min = self.pos + Vector3::new(-0.5, -0.5, -0.5);
    /// #         let max = self.pos + Vector3::new(0.5, 0.5, 0.5);
    /// #         AABB::with_bounds(min, max)
    /// #     }
    /// # }
    /// #
    /// # impl BHShape for UnitBox {
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
    /// let bvh = BVH::build(&mut shapes);
    /// let flat_bvh = bvh.flatten();
    /// ```
    pub fn flatten(&self) -> FlatBVH {
        self.flatten_custom(&|aabb, entry, exit, shape| FlatNode {
            aabb: *aabb,
            entry_index: entry,
            exit_index: exit,
            shape_index: shape,
        })
    }
}

impl BoundingHierarchy for FlatBVH {
    /// A [`FlatBVH`] is built from a regular [`BVH`] using the [`flatten`] method.
    ///
    /// [`FlatBVH`]: struct.FlatBVH.html
    /// [`BVH`]: ../bvh/struct.BVH.html
    ///
    fn build<T: BHShape>(shapes: &mut [T]) -> FlatBVH {
        let bvh = BVH::build(shapes);
        bvh.flatten()
    }

    /// Traverses a [`FlatBVH`] structure iteratively.
    ///
    /// [`FlatBVH`]: struct.FlatBVH.html
    ///
    /// # Examples
    ///
    /// ```
    /// use bvh::aabb::{AABB, Bounded};
    /// use bvh::bounding_hierarchy::BoundingHierarchy;
    /// use bvh::flat_bvh::FlatBVH;
    /// use bvh::{Point3, Vector3};
    /// use bvh::ray::Ray;
    /// # use bvh::bounding_hierarchy::BHShape;
    /// # pub struct UnitBox {
    /// #     pub id: i32,
    /// #     pub pos: Point3,
    /// #     node_index: usize,
    /// # }
    /// #
    /// # impl UnitBox {
    /// #     pub fn new(id: i32, pos: Point3) -> UnitBox {
    /// #         UnitBox {
    /// #             id: id,
    /// #             pos: pos,
    /// #             node_index: 0,
    /// #         }
    /// #     }
    /// # }
    /// #
    /// # impl Bounded for UnitBox {
    /// #     fn aabb(&self) -> AABB {
    /// #         let min = self.pos + Vector3::new(-0.5, -0.5, -0.5);
    /// #         let max = self.pos + Vector3::new(0.5, 0.5, 0.5);
    /// #         AABB::with_bounds(min, max)
    /// #     }
    /// # }
    /// #
    /// # impl BHShape for UnitBox {
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
    /// let flat_bvh = FlatBVH::build(&mut shapes);
    /// let hit_shapes = flat_bvh.traverse(&ray, &shapes);
    /// ```
    fn traverse<'a, T: Bounded>(&'a self, ray: &Ray, shapes: &'a [T]) -> Vec<&T> {
        let mut hit_shapes = Vec::new();
        let mut index = 0;

        // The traversal loop should terminate when `max_length` is set as the next node index.
        let max_length = self.len();

        // Iterate while the node index is valid.
        while index < max_length {
            let node = &self[index];

            if node.entry_index == u32::max_value() {
                // If the entry_index is MAX_UINT32, then it's a leaf node.
                let shape = &shapes[node.shape_index as usize];
                if ray.intersects_aabb(&shape.aabb()) {
                    hit_shapes.push(shape);
                }

                // Exit the current node.
                index = node.exit_index as usize;
            } else if ray.intersects_aabb(&node.aabb) {
                // If entry_index is not MAX_UINT32 and the AABB test passes, then
                // proceed to the node in entry_index (which goes down the bvh branch).
                index = node.entry_index as usize;
            } else {
                // If entry_index is not MAX_UINT32 and the AABB test fails, then
                // proceed to the node in exit_index (which defines the next untested partition).
                index = node.exit_index as usize;
            }
        }

        hit_shapes
    }

    /// Prints a textual representation of a [`FlatBVH`].
    ///
    /// [`FlatBVH`]: struct.FlatBVH.html
    ///
    fn pretty_print(&self) {
        for (i, node) in self.iter().enumerate() {
            println!(
                "{}\tentry {}\texit {}\tshape {}",
                i, node.entry_index, node.exit_index, node.shape_index
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::flat_bvh::FlatBVH;
    use crate::testbase::{build_some_bh, traverse_some_bh};

    #[test]
    /// Tests whether the building procedure succeeds in not failing.
    fn test_build_flat_bvh() {
        build_some_bh::<FlatBVH>();
    }

    #[test]
    /// Runs some primitive tests for intersections of a ray with a fixed scene given
    /// as a `FlatBVH`.
    fn test_traverse_flat_bvh() {
        traverse_some_bh::<FlatBVH>();
    }
}

#[cfg(all(feature = "bench", test))]
mod bench {
    use crate::bvh::BVH;
    use crate::flat_bvh::FlatBVH;

    use crate::testbase::{
        build_1200_triangles_bh, build_120k_triangles_bh, build_12k_triangles_bh, create_n_cubes,
        default_bounds, intersect_1200_triangles_bh, intersect_120k_triangles_bh,
        intersect_12k_triangles_bh,
    };

    #[bench]
    /// Benchmark the flattening of a BVH with 120,000 triangles.
    fn bench_flatten_120k_triangles_bvh(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        let bvh = BVH::build(&mut triangles);

        b.iter(|| {
            bvh.flatten();
        });
    }
    #[bench]
    /// Benchmark the construction of a `FlatBVH` with 1,200 triangles.
    fn bench_build_1200_triangles_flat_bvh(b: &mut ::test::Bencher) {
        build_1200_triangles_bh::<FlatBVH>(b);
    }

    #[bench]
    /// Benchmark the construction of a `FlatBVH` with 12,000 triangles.
    fn bench_build_12k_triangles_flat_bvh(b: &mut ::test::Bencher) {
        build_12k_triangles_bh::<FlatBVH>(b);
    }

    #[bench]
    /// Benchmark the construction of a `FlatBVH` with 120,000 triangles.
    fn bench_build_120k_triangles_flat_bvh(b: &mut ::test::Bencher) {
        build_120k_triangles_bh::<FlatBVH>(b);
    }

    #[bench]
    /// Benchmark intersecting 1,200 triangles using the recursive `FlatBVH`.
    fn bench_intersect_1200_triangles_flat_bvh(b: &mut ::test::Bencher) {
        intersect_1200_triangles_bh::<FlatBVH>(b);
    }

    #[bench]
    /// Benchmark intersecting 12,000 triangles using the recursive `FlatBVH`.
    fn bench_intersect_12k_triangles_flat_bvh(b: &mut ::test::Bencher) {
        intersect_12k_triangles_bh::<FlatBVH>(b);
    }

    #[bench]
    /// Benchmark intersecting 120,000 triangles using the recursive `FlatBVH`.
    fn bench_intersect_120k_triangles_flat_bvh(b: &mut ::test::Bencher) {
        intersect_120k_triangles_bh::<FlatBVH>(b);
    }
}
