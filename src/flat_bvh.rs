//! This module exports methods to flatten the `BVH` and traverse it iteratively.

use aabb::AABB;
use bvh::{BVH, BVHNode};
use ray::Ray;

/// A structure of a node of a flat [`BVH`]. The structure of the nodes allows for an
/// iterative traversal approach without the necessity to maintain a stack or queue.
///
/// [`BVH`]: struct.BVH.html
///
pub struct FlatNode {
    /// The [`AABB`] of the [`BVH`] node. Prior to testing the [`AABB`] bounds,
    /// the `entry_index` must be checked. In case the entry_index is [`u32::max_value()`],
    /// the [`AABB`] is undefined.
    ///
    /// [`AABB`]: ../aabb/struct.AABB.html
    /// [`BVH`]: struct.BVH.html
    /// [`u32::max_value()`]: https://doc.rust-lang.org/std/u32/constant.MAX.html
    ///
    aabb: AABB,

    /// The index of the `FlatNode` to jump to, if the [`AABB`] test is positive.
    /// If this value is [`u32::max_value()`] then the current node is a leaf node.
    /// Leaf nodes contain a shape index and an exit index. In leaf nodes the
    /// [`AABB`] is undefined.
    ///
    /// [`AABB`]: ../aabb/struct.AABB.html
    /// [`u32::max_value()`]: https://doc.rust-lang.org/std/u32/constant.MAX.html
    ///
    entry_index: u32,

    /// The index of the `FlatNode` to jump to, if the [`AABB`] test is negative.
    ///
    /// [`AABB`]: ../aabb/struct.AABB.html
    ///
    exit_index: u32,

    /// The index of the shape in the shapes array.
    shape_index: u32,
}

/// Prints a textual representation of a flat [`BVH`].
///
/// [`BVH`]: struct.BVH.html
///
pub fn pretty_print_flat_bvh(flat_nodes: &[FlatNode]) {
    for (i, node) in flat_nodes.iter().enumerate() {
        println!("{}\tentry {}\texit {}\tshape {}",
                 i,
                 node.entry_index,
                 node.exit_index,
                 node.shape_index);
    }
}

/// Traverses a flat [`BVH`] structure iteratively.
/// Returns a [`Vec`] of indices which are hit by `ray` with a high probability.
///
/// [`Vec`]: https://doc.rust-lang.org/std/vec/struct.Vec.html
/// [`BVH`]: struct.BVH.html
///
/// # Examples
///
/// ```
/// use bvh::aabb::{AABB, Bounded};
/// use bvh::bvh::BVH;
/// use bvh::flat_bvh::traverse_flat_bvh;
/// use bvh::nalgebra::{Point3, Vector3};
/// use bvh::ray::Ray;
///
/// # struct Sphere {
/// #     position: Point3<f32>,
/// #     radius: f32,
/// # }
/// #
/// # impl Bounded for Sphere {
/// #     fn aabb(&self) -> AABB {
/// #         let half_size = Vector3::new(self.radius, self.radius, self.radius);
/// #         let min = self.position - half_size;
/// #         let max = self.position + half_size;
/// #         AABB::with_bounds(min, max)
/// #     }
/// # }
/// #
/// # fn create_bounded_shapes() -> Vec<Sphere> {
/// #     let mut spheres = Vec::new();
/// #     for i in 0..1000u32 {
/// #         let position = Point3::new(i as f32, i as f32, i as f32);
/// #         let radius = (i % 10) as f32 + 1.0;
/// #         spheres.push(Sphere {
/// #             position: position,
/// #             radius: radius,
/// #         });
/// #     }
/// #     spheres
/// # }
/// #
/// let origin = Point3::new(0.0,0.0,0.0);
/// let direction = Vector3::new(1.0,0.0,0.0);
/// let ray = Ray::new(origin, direction);
/// let shapes = create_bounded_shapes();
/// let bvh = BVH::build(&shapes);
/// let flat_bvh = bvh.flatten();
/// let hit_shape_indices = traverse_flat_bvh(&ray, &flat_bvh);
/// ```
pub fn traverse_flat_bvh(ray: &Ray, flat_nodes: &[FlatNode]) -> Vec<usize> {
    let mut hit_shapes = Vec::new();
    let mut index = 0;

    // The traversal loop should terminate when `max_length` is set as the next node index
    let max_length = flat_nodes.len();

    // Iterate while the node index is valid
    while index < max_length {
        let node = &flat_nodes[index];

        if node.entry_index == u32::max_value() {
            // If the entry_index is MAX_UINT32, then it's a leaf node
            let shape_index = node.shape_index;
            hit_shapes.push(shape_index as usize);

            // Exit the current node
            index = node.exit_index as usize;
        } else if ray.intersects_aabb(&node.aabb) {
            // If entry_index is not MAX_UINT32 and the AABB test passes, then
            // proceed to the node in entry_index (which goes down the bvh branch)
            index = node.entry_index as usize;
        } else {
            // If entry_index is not MAX_UINT32 and the AABB test fails, then
            // proceed to the node in exit_index (which defines the next untested partition)
            index = node.exit_index as usize;
        }
    }

    hit_shapes
}

impl BVHNode {
    /// Flattens the [`BVH`], so that it can be traversed in an iterative manner.
    /// The iterative traverse procedure is implemented in [`traverse_flat_bvh()`].
    ///
    /// [`BVH`]: struct.BVH.html
    /// [`traverse_flat_bvh()`]: method.traverse_flat_bvh.html
    ///
    pub fn flatten(&self, nodes: &Vec<BVHNode>, vec: &mut Vec<FlatNode>, next_free: usize) -> usize {
        match *self {
            BVHNode::Node { child_l_aabb, child_l, child_r_aabb, child_r, .. } => {
                // Create the enclosing node for the left subtree
                vec.push(FlatNode {
                    aabb: child_l_aabb,
                    entry_index: (next_free + 1) as u32,
                    exit_index: 0,
                    shape_index: u32::max_value(),
                });

                // Create the flat left subtree and update the exit index in the enclosing node
                let index_after_child_l = nodes[child_l].flatten(nodes, vec, next_free + 1);
                vec[next_free as usize].exit_index = index_after_child_l as u32;

                // Create the enclosing node for the right subtree
                vec.push(FlatNode {
                    aabb: child_r_aabb,
                    entry_index: (index_after_child_l + 1) as u32,
                    exit_index: 0,
                    shape_index: u32::max_value(),
                });

                // Create the flat right subtree and update the exit index in the enclosing node
                let index_after_child_r = nodes[child_r].flatten(nodes, vec, index_after_child_l + 1);
                vec[index_after_child_l as usize].exit_index = index_after_child_r as u32;

                index_after_child_r
            }
            BVHNode::Leaf { shape, .. } => {
                let mut next_shape = next_free;

                next_shape += 1;

                // Create the flat leaf node
                vec.push(FlatNode {
                    aabb: AABB::empty(),
                    entry_index: u32::max_value(),
                    exit_index: next_shape as u32,
                    shape_index: shape as u32,
                });

                next_shape
            }
            BVHNode::Dummy => panic!("Found dummy node while flattening BVH.")
        }
    }

    /// Creates a flat node from a BVH inner node and its AABB. Returns the next free index.
    /// TODO: change the algorithm which pushes `FlatNode`s to a vector to not use indices this
    /// much. Implement an algorithm which writes directly to a writable slice.
    fn create_flat_branch<F, FNodeType>(&self,
                                        nodes: &Vec<BVHNode>,
                                        this_aabb: &AABB,
                                        vec: &mut Vec<FNodeType>,
                                        next_free: usize,
                                        constructor: &F)
                                        -> usize
        where F: Fn(&AABB, u32, u32, u32) -> FNodeType
    {
        // Create dummy node
        let dummy = constructor(&AABB::empty(), 0, 0, 0);
        vec.push(dummy);
        assert!(vec.len() - 1 == next_free);

        // Create subtree
        let index_after_subtree = self.flatten_custom(nodes, vec, next_free + 1, constructor);

        // Replace dummy node by actual node with the entry index pointing to the subtree
        // and the exit index pointing to the next node after the subtree
        let navigator_node = constructor(this_aabb,
                                         (next_free + 1) as u32,
                                         index_after_subtree as u32,
                                         u32::max_value());
        vec[next_free] = navigator_node;
        index_after_subtree
    }

    /// Flattens the [`BVH`], so that it can be traversed in an iterative manner.
    /// The iterative traverse procedure is implemented in [`traverse_flat_bvh`].
    /// This method constructs custom flat nodes using the `constructor`.
    ///
    /// [`BVH`]: struct.BVH.html
    /// [`traverse_flat_bvh`]: method.traverse_flat_bvh.html
    ///
    pub fn flatten_custom<F, FNodeType>(&self,
                                        nodes: &Vec<BVHNode>,
                                        vec: &mut Vec<FNodeType>,
                                        next_free: usize,
                                        constructor: &F)
                                        -> usize
        where F: Fn(&AABB, u32, u32, u32) -> FNodeType
    {
        match *self {
            BVHNode::Node { ref child_l_aabb, child_l, ref child_r_aabb, child_r, .. } => {
                let index_after_child_l =
                    nodes[child_l].create_flat_branch(nodes, child_l_aabb, vec, next_free, constructor);
                nodes[child_r].create_flat_branch(nodes, child_r_aabb, vec, index_after_child_l, constructor)
            }

            BVHNode::Leaf { shape, .. } => {
                let mut next_shape = next_free;
                next_shape += 1;
                let leaf_node = constructor(&AABB::empty(),
                                            u32::max_value(),
                                            next_shape as u32,
                                            shape as u32);
                vec.push(leaf_node);

                next_shape
            }

            BVHNode::Dummy => panic!("Found dummy node while flattening BVH.")
        }
    }
}

impl BVH {
    /// Flattens the [`BVH`] so that it can be traversed iteratively.
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    /// # Examples
    ///
    /// ```
    /// use bvh::aabb::{AABB, Bounded};
    /// use bvh::bvh::BVH;
    /// use bvh::nalgebra::{Point3, Vector3};
    /// use bvh::ray::Ray;
    ///
    /// # struct Sphere {
    /// #     position: Point3<f32>,
    /// #     radius: f32,
    /// # }
    /// #
    /// # impl Bounded for Sphere {
    /// #     fn aabb(&self) -> AABB {
    /// #         let half_size = Vector3::new(self.radius, self.radius, self.radius);
    /// #         let min = self.position - half_size;
    /// #         let max = self.position + half_size;
    /// #         AABB::with_bounds(min, max)
    /// #     }
    /// # }
    /// #
    /// # fn create_bounded_shapes() -> Vec<Sphere> {
    /// #     let mut spheres = Vec::new();
    /// #     for i in 0..1000u32 {
    /// #         let position = Point3::new(i as f32, i as f32, i as f32);
    /// #         let radius = (i % 10) as f32 + 1.0;
    /// #         spheres.push(Sphere {
    /// #             position: position,
    /// #             radius: radius,
    /// #         });
    /// #     }
    /// #     spheres
    /// # }
    /// #
    /// let origin = Point3::new(0.0,0.0,0.0);
    /// let direction = Vector3::new(1.0,0.0,0.0);
    /// let ray = Ray::new(origin, direction);
    /// let shapes = create_bounded_shapes();
    /// let bvh = BVH::build(&shapes);
    /// let flat_bvh = bvh.flatten();
    /// ```
    pub fn flatten(&self) -> Vec<FlatNode> {
        let mut vec = Vec::new();
        self.nodes[0].flatten(&self.nodes, &mut vec, 0);
        vec
    }

    /// Flattens the [`BVH`] so that it can be traversed iteratively.
    /// Constructs the flat nodes using the supplied function.
    /// This function can be used, when the flat bvh nodes should be of some particular
    /// non-default structure.
    /// The `constructor` is fed the following arguments in this order:
    ///
    /// 1.0 &AABB: The enclosing `AABB`
    /// 2.0 u32: The index of the nested node
    /// 3.0 u32: The exit index
    /// 4.0 u32: The shape index
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    /// # Examples
    ///
    /// ```
    /// use bvh::aabb::{AABB, Bounded};
    /// use bvh::bvh::BVH;
    /// use bvh::nalgebra::{Point3, Vector3};
    /// use bvh::ray::Ray;
    ///
    /// # struct Sphere {
    /// #     position: Point3<f32>,
    /// #     radius: f32,
    /// # }
    /// #
    /// # impl Bounded for Sphere {
    /// #     fn aabb(&self) -> AABB {
    /// #         let half_size = Vector3::new(self.radius, self.radius, self.radius);
    /// #         let min = self.position - half_size;
    /// #         let max = self.position + half_size;
    /// #         AABB::with_bounds(min, max)
    /// #     }
    /// # }
    /// #
    /// # fn create_bounded_shapes() -> Vec<Sphere> {
    /// #     let mut spheres = Vec::new();
    /// #     for i in 0..1000u32 {
    /// #         let position = Point3::new(i as f32, i as f32, i as f32);
    /// #         let radius = (i % 10) as f32 + 1.0;
    /// #         spheres.push(Sphere {
    /// #             position: position,
    /// #             radius: radius,
    /// #         });
    /// #     }
    /// #     spheres
    /// # }
    /// #
    /// struct CustomStruct {
    ///     aabb: AABB,
    ///     entry_index: u32,
    ///     exit_index: u32,
    ///     shape_index: u32,
    /// }
    ///
    /// fn custom_struct_constructor(aabb: &AABB,
    ///                              entry_index: u32,
    ///                              exit_index: u32,
    ///                              shape_index: u32)
    ///                              -> CustomStruct {
    ///     CustomStruct {
    ///         aabb: *aabb,
    ///         entry_index: entry_index,
    ///         exit_index: exit_index,
    ///         shape_index: shape_index,
    ///     }
    /// }
    ///
    /// let shapes = create_bounded_shapes();
    /// let bvh = BVH::build(&shapes);
    /// let custom_flat_bvh = bvh.flatten_custom(&custom_struct_constructor);
    /// ```
    pub fn flatten_custom<F, FNodeType>(&self, constructor: &F) -> Vec<FNodeType>
        where F: Fn(&AABB, u32, u32, u32) -> FNodeType
    {
        let mut vec = Vec::new();
        self.nodes[0].flatten_custom(&self.nodes, &mut vec, 0, constructor);
        vec
    }
}

#[cfg(test)]
mod tests {
    use aabb::{AABB, Bounded};
    use bvh::BVH;
    use bvh::tests::{XBox, build_some_bvh, create_n_cubes, create_ray};
    use flat_bvh::{traverse_flat_bvh, FlatNode};
    use nalgebra::{Point3, Vector3};
    use std::collections::HashSet;
    use ray::Ray;

    #[test]
    /// Builds and flattens a BVH. Tests whether the `flatten` procedure succeeds.
    fn test_flatten() {
        let (_, bvh) = build_some_bvh();
        bvh.flatten();
    }

    #[test]
    /// Runs some primitive tests for intersections of a ray with
    /// a fixed scene given as a flat BVH.
    fn test_traverse_flat_bvh() {
        let (shapes, bvh) = build_some_bvh();
        let flat_bvh = bvh.flatten();

        fn test_ray(ray: &Ray, flat_bvh: &[FlatNode], shapes: &[XBox]) -> Vec<usize> {
            let hit_shapes = traverse_flat_bvh(ray, flat_bvh);
            for (index, shape) in shapes.iter().enumerate() {
                if !hit_shapes.contains(&index) {
                    assert!(!ray.intersects_aabb(&shape.aabb()));
                }
            }
            hit_shapes
        }

        // Define a ray which traverses the x-axis from afar
        let position_1 = Point3::new(-1000.0, 0.0, 0.0);
        let direction_1 = Vector3::new(1.0, 0.0, 0.0);
        let ray_1 = Ray::new(position_1, direction_1);
        let hit_shapes_1 = test_ray(&ray_1, &flat_bvh, &shapes);
        for i in 0..21 {
            assert!(hit_shapes_1.contains(&i));
        }

        // Define a ray which traverses the y-axis from afar
        let position_2 = Point3::new(1.0, -1000.0, 0.0);
        let direction_2 = Vector3::new(0.0, 1.0, 0.0);
        let ray_2 = Ray::new(position_2, direction_2);
        let hit_shapes_2 = test_ray(&ray_2, &flat_bvh, &shapes);
        assert!(hit_shapes_2.contains(&11));

        // Define a ray which intersects the x-axis diagonally
        let position_3 = Point3::new(6.0, 0.5, 0.0);
        let direction_3 = Vector3::new(-2.0, -1.0, 0.0);
        let ray_3 = Ray::new(position_3, direction_3);

        // It should hit exactly three boxes
        let hit_shapes_3 = test_ray(&ray_3, &flat_bvh, &shapes);
        assert!(hit_shapes_3.len() == 3);
        let mut xs_3 = HashSet::new();
        for shape in &hit_shapes_3 {
            xs_3.insert(shapes[*shape].x);
        }
        assert!(xs_3.contains(&6));
        assert!(xs_3.contains(&5));
        assert!(xs_3.contains(&4));
    }

    #[test]
    /// Test whether the `flatten_custom` produces a flat `BVH` with the same relative structure
    /// as `flatten`.
    fn test_compare_default_and_custom_flat_bvh() {
        fn custom_constructor(aabb: &AABB,
                              entry_index: u32,
                              exit_index: u32,
                              shape_index: u32)
                              -> FlatNode {
            FlatNode {
                aabb: *aabb,
                entry_index: entry_index,
                exit_index: exit_index,
                shape_index: shape_index,
            }
        }

        // Generate a BVH and flatten it defaultly, and using a custom constructor
        let triangles = create_n_cubes(1_000);
        let bvh = BVH::build(&triangles);
        let flat_bvh = bvh.flatten();
        let flat_bvh_custom = bvh.flatten_custom(&custom_constructor);

        // It should produce the same structure in both cases
        for (default_node, custom_node) in flat_bvh.iter().zip(flat_bvh_custom.iter()) {
            assert_eq!(default_node.entry_index, custom_node.entry_index);
            assert_eq!(default_node.exit_index, custom_node.exit_index);
            assert_eq!(default_node.shape_index, custom_node.shape_index);
        }
    }

    #[bench]
    /// Benchmark the flattening of a BVH with 120,000 triangles.
    fn bench_flatten_120k_triangles_bvh(b: &mut ::test::Bencher) {
        let triangles = create_n_cubes(10_000);
        let bvh = BVH::build(&triangles);

        b.iter(|| {
            bvh.flatten();
        });
    }

    #[bench]
    /// Benchmark intersecting 120,000 triangles using the recursive BVH.
    fn bench_intersect_120k_triangles_bvh_flat(b: &mut ::test::Bencher) {
        let triangles = create_n_cubes(10_000);
        let bvh = BVH::build(&triangles);
        let flat_bvh = bvh.flatten();
        let mut seed = 0;

        b.iter(|| {
            let ray = create_ray(&mut seed);

            // Traverse the flat BVH
            let hits = traverse_flat_bvh(&ray, &flat_bvh);

            // Traverse the resulting list of positive AABB tests
            for index in &hits {
                let triangle = &triangles[*index];
                ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c);
            }
        });
    }
}
