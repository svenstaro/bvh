//! This module defines [`BVH`] and [`BVHNode`] and functions for building and traversing it.
//!
//! [`BVH`]: struct.BVH.html
//! [`BVHNode`]: struct.BVHNode.html
//!

use EPSILON;
use aabb::{AABB, Bounded};
use bounding_hierarchy::{BoundingHierarchy, BHShape};
use ray::Ray;
use std::f32;
use std::iter::repeat;

/// The [`BVHNode`] enum that describes a node in a [`BVH`].
/// It's either a leaf node and references a shape (by holding its index)
/// or a regular node that has two child nodes.
/// The non-leaf node stores the [`AABB`]s of its children.
///
/// [`AABB`]: ../aabb/struct.AABB.html
/// [`BVH`]: struct.BVH.html
/// [`BVH`]: struct.BVHNode.html
///
#[derive(Debug, Copy, Clone)]
pub enum BVHNode {
    /// Leaf node.
    Leaf {
        /// The node's parent.
        parent_index: usize,

        /// The node's depth.
        depth: u32,

        /// The shape contained in this leaf.
        shape_index: usize,
    },
    /// Inner node.
    Node {
        /// The node's parent.
        parent_index: usize,

        /// The node's depth.
        depth: u32,

        /// Index of the left subtree's root node.
        child_l_index: usize,

        /// The convex hull of the shapes' `AABB`s in child_l.
        child_l_aabb: AABB,

        /// Index of the right subtree's root node.
        child_r_index: usize,

        /// The convex hull of the shapes' `AABB`s in child_r.
        child_r_aabb: AABB,
    },
}

impl PartialEq for BVHNode {
    // TODO Consider also comparing AABBs
    fn eq(&self, other: &BVHNode) -> bool {
        match (self, other) {
            (&BVHNode::Node {
                  parent_index: self_parent_index,
                  depth: self_depth,
                  child_l_index: self_child_l_index,
                  child_r_index: self_child_r_index,
                  ..
              },
             &BVHNode::Node {
                  parent_index: other_parent_index,
                  depth: other_depth,
                  child_l_index: other_child_l_index,
                  child_r_index: other_child_r_index,
                  ..
              }) => {
                self_parent_index == other_parent_index && self_depth == other_depth &&
                self_child_l_index == other_child_l_index &&
                self_child_r_index == other_child_r_index
            }
            (&BVHNode::Leaf {
                  parent_index: self_parent_index,
                  depth: self_depth,
                  shape_index: self_shape_index,
              },
             &BVHNode::Leaf {
                  parent_index: other_parent_index,
                  depth: other_depth,
                  shape_index: other_shape_index,
              }) => {
                self_parent_index == other_parent_index && self_depth == other_depth &&
                self_shape_index == other_shape_index
            }
            _ => false,
        }
    }
}

impl BVHNode {
    /// Returns the index of the parent node.
    pub fn parent(&self) -> usize {
        match *self {
            BVHNode::Node { parent_index, .. } |
            BVHNode::Leaf { parent_index, .. } => parent_index,
        }
    }

    /// Returns a mutable reference to the parent node index.
    pub fn parent_mut(&mut self) -> &mut usize {
        match *self {
            mut BVHNode::Node { ref mut parent_index, .. } |
            mut BVHNode::Leaf { ref mut parent_index, .. } => parent_index,
        }
    }

    /// Returns the index of the left child node.
    pub fn child_l(&self) -> usize {
        match *self {
            BVHNode::Node { child_l_index, .. } => child_l_index,
            _ => panic!("Tried to get the left child of a leaf node."),
        }
    }

    /// Returns the `AABB` of the right child node.
    pub fn child_l_aabb(&self) -> AABB {
        match *self {
            BVHNode::Node { child_l_aabb, .. } => child_l_aabb,
            _ => panic!(),
        }
    }

    /// Returns a mutable reference to the `AABB` of the left child node.
    pub fn child_l_aabb_mut(&mut self) -> &mut AABB {
        match *self {
            mut BVHNode::Node { ref mut child_l_aabb, .. } => child_l_aabb,
            _ => panic!("Tried to get the left child's `AABB` of a leaf node."),
        }
    }

    /// Returns the index of the right child node.
    pub fn child_r(&self) -> usize {
        match *self {
            BVHNode::Node { child_r_index, .. } => child_r_index,
            _ => panic!("Tried to get the right child of a leaf node."),
        }
    }

    /// Returns the `AABB` of the right child node.
    pub fn child_r_aabb(&self) -> AABB {
        match *self {
            BVHNode::Node { child_r_aabb, .. } => child_r_aabb,
            _ => panic!(),
        }
    }

    /// Returns a mutable reference to the `AABB` of the right child node.
    pub fn child_r_aabb_mut(&mut self) -> &mut AABB {
        match *self {
            mut BVHNode::Node { ref mut child_r_aabb, .. } => child_r_aabb,
            _ => panic!("Tried to get the right child's `AABB` of a leaf node."),
        }
    }

    /// Returns the depth of the node. The root node has depth `0`.
    pub fn depth(&self) -> u32 {
        match *self {
            BVHNode::Node { depth, .. } |
            BVHNode::Leaf { depth, .. } => depth,
        }
    }

    /// Gets the `AABB` for a `BVHNode`.
    /// Returns the shape's `AABB` for leaves, and the joined `AABB` of
    /// the two children's `AABB`s for non-leaves.
    pub fn get_node_aabb<Shape: BHShape>(&self, shapes: &[Shape]) -> AABB {
        match *self {
            BVHNode::Node {
                 child_l_aabb,
                 child_r_aabb,
                 ..
             } => child_l_aabb.join(&child_r_aabb),
            BVHNode::Leaf { shape_index, .. } => shapes[shape_index].aabb(),
        }
    }

    /// The build function sometimes needs to add nodes while their data is not available yet.
    /// A dummy cerated by this function serves the purpose of being changed later on.
    fn create_dummy() -> BVHNode {
        BVHNode::Leaf {
            parent_index: 0,
            depth: 0,
            shape_index: 0,
        }
    }

    /// Builds a [`BVHNode`] recursively using SAH partitioning.
    /// Returns the index of the new node in the nodes vector.
    ///
    /// [`BVHNode`]: enum.BVHNode.html
    ///
    pub fn build<T: BHShape>(shapes: &mut [T],
                             indices: Vec<usize>,
                             nodes: &mut Vec<BVHNode>,
                             parent_index: usize,
                             depth: u32)
                             -> usize {
        // Helper function to accumulate the AABB joint and the centroids AABB
        fn grow_convex_hull(convex_hull: (AABB, AABB), shape_aabb: &AABB) -> (AABB, AABB) {
            let center = &shape_aabb.center();
            let convex_hull_aabbs = &convex_hull.0;
            let convex_hull_centroids = &convex_hull.1;
            (convex_hull_aabbs.join(shape_aabb), convex_hull_centroids.grow(center))
        }

        let mut convex_hull = Default::default();
        for index in &indices {
            convex_hull = grow_convex_hull(convex_hull, &shapes[*index].aabb());
        }
        let (aabb_bounds, centroid_bounds) = convex_hull;

        // If there is only one element left, don't split anymore
        if indices.len() == 1 {
            let shape_index = indices[0];
            let node_index = nodes.len();
            nodes.push(BVHNode::Leaf {
                           parent_index: parent_index,
                           depth: depth,
                           shape_index: shape_index,
                       });
            // Let the shape know the index of the node that represents it
            shapes[shape_index].set_bh_node_index(node_index);
            return node_index;
        }

        // Find the axis along which the shapes are spread the most
        let split_axis = centroid_bounds.largest_axis();
        let split_axis_size = centroid_bounds.max[split_axis] - centroid_bounds.min[split_axis];

        /// Defines a Bucket utility object.
        #[derive(Copy, Clone)]
        struct Bucket {
            size: usize,
            aabb: AABB,
        }

        impl Bucket {
            /// Returns an empty bucket.
            fn empty() -> Bucket {
                Bucket {
                    size: 0,
                    aabb: AABB::empty(),
                }
            }

            /// Extends this `Bucket` by the given `AABB`.
            fn add_aabb(&mut self, aabb: &AABB) {
                self.size += 1;
                self.aabb = self.aabb.join(aabb);
            }
        }

        /// Joins two `Bucket`s.
        fn join_bucket(a: Bucket, b: &Bucket) -> Bucket {
            Bucket {
                size: a.size + b.size,
                aabb: a.aabb.join(&b.aabb),
            }
        }

        // Create six buckets, and six index assignment vectors
        const NUM_BUCKETS: usize = 6;
        let mut buckets = [Bucket::empty(); NUM_BUCKETS];
        let mut bucket_assignments: [Vec<usize>; NUM_BUCKETS] = Default::default();

        if split_axis_size < EPSILON {
            // Spread the remaining shapes evenly across buckets,
            // instead of using a heuristic
            for (counter, idx) in (&indices).iter().enumerate() {
                let shape = &shapes[*idx];
                let shape_aabb = shape.aabb();

                let bucket_num = counter % NUM_BUCKETS;

                buckets[bucket_num].add_aabb(&shape_aabb);
                bucket_assignments[bucket_num].push(*idx);
            }
        } else {
            // Assign each shape to a bucket
            for idx in &indices {
                let shape = &shapes[*idx];
                let shape_aabb = shape.aabb();
                let shape_center = shape_aabb.center();

                // Get the relative position of the shape centroid [0.0..1.0]
                let bucket_num_relative =
                    (shape_center[split_axis] - centroid_bounds.min[split_axis]) / split_axis_size;

                // Convert that to the actual `Bucket` number
                let bucket_num = (bucket_num_relative * (NUM_BUCKETS as f32 - 0.01)) as usize;

                // Extend the selected `Bucket` and add the index to the actual bucket
                buckets[bucket_num].add_aabb(&shape_aabb);
                bucket_assignments[bucket_num].push(*idx);
            }
        }

        // Compute the costs for each configuration and
        // select the configuration with the minimal costs
        let mut min_bucket = 0;
        let mut min_cost = f32::INFINITY;
        let mut child_l_aabb = AABB::empty();
        let mut child_r_aabb = AABB::empty();
        for i in 0..(NUM_BUCKETS - 1) {
            let (l_buckets, r_buckets) = buckets.split_at(i + 1);
            let child_l = l_buckets.iter().fold(Bucket::empty(), join_bucket);
            let child_r = r_buckets.iter().fold(Bucket::empty(), join_bucket);

            let cost = (child_l.size as f32 * child_l.aabb.surface_area() +
                        child_r.size as f32 * child_r.aabb.surface_area()) /
                       aabb_bounds.surface_area();

            if cost < min_cost {
                min_bucket = i;
                min_cost = cost;
                child_l_aabb = child_l.aabb;
                child_r_aabb = child_r.aabb;
            }
        }

        fn join_vectors(vectors: &mut [Vec<usize>]) -> Vec<usize> {
            let mut result = Vec::new();
            for mut vector in vectors.iter_mut() {
                result.append(&mut vector);
            }
            result
        }

        // Join together all index buckets, and proceed recursively
        let (l_assignments, r_assignments) = bucket_assignments.split_at_mut(min_bucket + 1);
        let child_l_indices = join_vectors(l_assignments);
        let child_r_indices = join_vectors(r_assignments);

        let node_index = nodes.len();
        nodes.push(BVHNode::create_dummy());

        let child_l_index = BVHNode::build(shapes, child_l_indices, nodes, node_index, depth + 1);
        let child_r_index = BVHNode::build(shapes, child_r_indices, nodes, node_index, depth + 1);

        // Construct the actual data structure
        nodes[node_index] = BVHNode::Node {
            parent_index: parent_index,
            depth: depth,
            child_l_aabb: child_l_aabb,
            child_l_index: child_l_index,
            child_r_aabb: child_r_aabb,
            child_r_index: child_r_index,
        };

        node_index
    }

    /// Traverses the [`BVH`] recursively and returns all shapes whose [`AABB`] is
    /// intersected by the given [`Ray`].
    ///
    /// [`AABB`]: ../aabb/struct.AABB.html
    /// [`BVH`]: struct.BVH.html
    /// [`Ray`]: ../ray/struct.Ray.html
    ///
    pub fn traverse_recursive(nodes: &Vec<BVHNode>,
                              node_index: usize,
                              ray: &Ray,
                              indices: &mut Vec<usize>) {
        match &nodes[node_index] {
            &BVHNode::Node {
                 ref child_l_aabb,
                 child_l_index,
                 ref child_r_aabb,
                 child_r_index,
                 ..
             } => {
                if ray.intersects_aabb(child_l_aabb) {
                    BVHNode::traverse_recursive(nodes, child_l_index, ray, indices);
                }
                if ray.intersects_aabb(child_r_aabb) {
                    BVHNode::traverse_recursive(nodes, child_r_index, ray, indices);
                }
            }
            &BVHNode::Leaf { shape_index, .. } => {
                indices.push(shape_index);
            }
        }
    }
}

/// The [`BVH`] data structure. Contains the list of [`BVHNode`]s.
///
/// [`BVH`]: struct.BVH.html
///
pub struct BVH {
    /// The list of nodes of the [`BVH`].
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    pub nodes: Vec<BVHNode>,
}

impl BVH {
    /// Creates a new [`BVH`] from the `shapes` slice.
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    pub fn build<Shape: BHShape>(shapes: &mut [Shape]) -> BVH {
        let indices = (0..shapes.len()).collect::<Vec<usize>>();
        let expected_node_count = shapes.len() * 2;
        let mut nodes = Vec::with_capacity(expected_node_count);
        BVHNode::build(shapes, indices, &mut nodes, 0, 0);
        BVH { nodes: nodes }
    }

    /// Traverses the [`BVH`].
    /// Returns a subset of `shapes`, in which the [`AABB`]s of the elements were hit by `ray`.
    ///
    /// [`BVH`]: struct.BVH.html
    /// [`AABB`]: ../aabb/struct.AABB.html
    ///
    pub fn traverse<'a, Shape: Bounded>(&'a self, ray: &Ray, shapes: &'a [Shape]) -> Vec<&Shape> {
        let mut indices = Vec::new();
        BVHNode::traverse_recursive(&self.nodes, 0, ray, &mut indices);
        indices
            .iter()
            .map(|index| &shapes[*index])
            .collect::<Vec<_>>()
    }

    /// Prints the [`BVH`] in a tree-like visualization.
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    pub fn pretty_print(&self) {
        let nodes = &self.nodes;
        fn print_node(nodes: &Vec<BVHNode>, node_index: usize) {
            match &nodes[node_index] {
                &BVHNode::Node {
                     child_l_index,
                     child_r_index,
                     depth,
                     child_l_aabb,
                     child_r_aabb,
                     ..
                 } => {
                    let padding: String = repeat(" ").take(depth as usize).collect();
                    println!("{}child_l {:?}", padding, child_l_aabb);
                    print_node(nodes, child_l_index);
                    println!("{}child_r {:?}", padding, child_r_aabb);
                    print_node(nodes, child_r_index);
                }
                &BVHNode::Leaf { shape_index, depth, .. } => {
                    let padding: String = repeat(" ").take(depth as usize).collect();
                    println!("{}shape\t{:?}", padding, shape_index);
                }
            }
        }
        print_node(nodes, 0);
    }
}

impl BoundingHierarchy for BVH {
    fn build<Shape: BHShape>(shapes: &mut [Shape]) -> BVH {
        BVH::build(shapes)
    }

    fn traverse<'a, Shape: Bounded>(&'a self, ray: &Ray, shapes: &'a [Shape]) -> Vec<&Shape> {
        self.traverse(ray, shapes)
    }

    fn pretty_print(&self) {
        self.pretty_print();
    }
}

#[cfg(test)]
pub mod tests {
    use bvh::BVH;
    use testbase::{build_some_bh, traverse_some_bh, build_1200_triangles_bh,
                   build_12k_triangles_bh, build_120k_triangles_bh, intersect_1200_triangles_bh,
                   intersect_12k_triangles_bh, intersect_120k_triangles_bh};

    #[test]
    /// Tests whether the building procedure succeeds in not failing.
    fn test_build_bvh() {
        build_some_bh::<BVH>();
    }

    #[test]
    /// Runs some primitive tests for intersections of a ray with a fixed scene given as a BVH.
    fn test_traverse_bvh() {
        traverse_some_bh::<BVH>();
    }

    #[bench]
    /// Benchmark the construction of a `BVH` with 1,200 triangles.
    fn bench_build_1200_triangles_bvh(mut b: &mut ::test::Bencher) {
        build_1200_triangles_bh::<BVH>(&mut b);
    }

    #[bench]
    /// Benchmark the construction of a `BVH` with 12,000 triangles.
    fn bench_build_12k_triangles_bvh(mut b: &mut ::test::Bencher) {
        build_12k_triangles_bh::<BVH>(&mut b);
    }

    #[bench]
    /// Benchmark the construction of a `BVH` with 120,000 triangles.
    fn bench_build_120k_triangles_bvh(mut b: &mut ::test::Bencher) {
        build_120k_triangles_bh::<BVH>(&mut b);
    }

    #[bench]
    /// Benchmark intersecting 1,200 triangles using the recursive `BVH`.
    fn bench_intersect_1200_triangles_bvh(mut b: &mut ::test::Bencher) {
        intersect_1200_triangles_bh::<BVH>(&mut b);
    }

    #[bench]
    /// Benchmark intersecting 12,000 triangles using the recursive `BVH`.
    fn bench_intersect_12k_triangles_bvh(mut b: &mut ::test::Bencher) {
        intersect_12k_triangles_bh::<BVH>(&mut b);
    }

    #[bench]
    /// Benchmark intersecting 120,000 triangles using the recursive `BVH`.
    fn bench_intersect_120k_triangles_bvh(mut b: &mut ::test::Bencher) {
        intersect_120k_triangles_bh::<BVH>(&mut b);
    }
}
