//! This module defines [`BVH`] and [`BVHNode`] and functions for building and traversing it.
//!
//! [`BVH`]: struct.BVH.html
//! [`BVHNode`]: struct.BVHNode.html
//!

use crate::aabb::{Bounded, AABB};
use crate::axis::Axis;
use crate::bounding_hierarchy::{BHShape, BoundingHierarchy, IntersectionAABB};
use crate::bvh::iter::BVHTraverseIterator;
use crate::utils::{joint_aabb_of_shapes, Bucket};
use crate::EPSILON;
use crate::{Point3, Real};
use rayon::prelude::*;

use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::iter::repeat;
use std::slice;

const NUM_BUCKETS: usize = 6;

thread_local! {
    pub static BUCKETS: RefCell<[Vec<usize>; NUM_BUCKETS]> = RefCell::new(Default::default());
}

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
#[cfg_attr(feature = "serde_impls", derive(serde::Serialize, serde::Deserialize))]
#[allow(clippy::upper_case_acronyms)]
pub enum BVHNode {
    /// Leaf node.
    Leaf {
        /// The node's parent.
        parent_index: usize,

        /// The shape contained in this leaf.
        shape_index: usize,
    },
    /// Inner node.
    Node {
        /// The node's parent.
        parent_index: usize,

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
            (
                &BVHNode::Node {
                    parent_index: self_parent_index,
                    child_l_index: self_child_l_index,
                    child_r_index: self_child_r_index,
                    ..
                },
                &BVHNode::Node {
                    parent_index: other_parent_index,
                    child_l_index: other_child_l_index,
                    child_r_index: other_child_r_index,
                    ..
                },
            ) => {
                self_parent_index == other_parent_index
                    && self_child_l_index == other_child_l_index
                    && self_child_r_index == other_child_r_index
            }
            (
                &BVHNode::Leaf {
                    parent_index: self_parent_index,
                    shape_index: self_shape_index,
                },
                &BVHNode::Leaf {
                    parent_index: other_parent_index,
                    shape_index: other_shape_index,
                },
            ) => self_parent_index == other_parent_index && self_shape_index == other_shape_index,
            _ => false,
        }
    }
}

impl BVHNode {
    /// Returns the index of the parent node.
    pub fn parent(&self) -> usize {
        match *self {
            BVHNode::Node { parent_index, .. } | BVHNode::Leaf { parent_index, .. } => parent_index,
        }
    }

    /// Returns a mutable reference to the parent node index.
    pub fn parent_mut(&mut self) -> &mut usize {
        match *self {
            BVHNode::Node {
                ref mut parent_index,
                ..
            }
            | BVHNode::Leaf {
                ref mut parent_index,
                ..
            } => parent_index,
        }
    }

    /// Returns the index of the left child node.
    pub fn child_l(&self) -> usize {
        match *self {
            BVHNode::Node { child_l_index, .. } => child_l_index,
            _ => panic!("Tried to get the left child of a leaf node."),
        }
    }
    /// Returns the index of the left child node.
    pub fn child_l_mut(&mut self) -> &mut usize {
        match *self {
            BVHNode::Node {
                ref mut child_l_index,
                ..
            } => child_l_index,
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
            BVHNode::Node {
                ref mut child_l_aabb,
                ..
            } => child_l_aabb,
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

    /// Returns the index of the right child node.
    pub fn child_r_mut(&mut self) -> &mut usize {
        match *self {
            BVHNode::Node {
                ref mut child_r_index,
                ..
            } => child_r_index,
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
            BVHNode::Node {
                ref mut child_r_aabb,
                ..
            } => child_r_aabb,
            _ => panic!("Tried to get the right child's `AABB` of a leaf node."),
        }
    }

    /// Returns the depth of the node. The root node has depth `0`.
    pub fn depth(&self, nodes: &[BVHNode]) -> u32 {
        let parent_i = self.parent();
        if parent_i == 0 {
            if nodes[parent_i].eq(&self) {
                return 0;
            }
        }
        return 1 + nodes[parent_i].depth(nodes);
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

    /// Returns the index of the shape contained within the node if is a leaf,
    /// or `None` if it is an interior node.
    pub fn shape_index(&self) -> Option<usize> {
        match *self {
            BVHNode::Leaf { shape_index, .. } => Some(shape_index),
            _ => None,
        }
    }

    /// Returns the index of the shape contained within the node if is a leaf,
    /// or `None` if it is an interior node.
    pub fn shape_index_mut(&mut self) -> Option<&mut usize> {
        match *self {
            BVHNode::Leaf {
                ref mut shape_index,
                ..
            } => Some(shape_index),
            _ => None,
        }
    }

    /// The build function sometimes needs to add nodes while their data is not available yet.
    /// A dummy cerated by this function serves the purpose of being changed later on.
    fn create_dummy() -> BVHNode {
        BVHNode::Leaf {
            parent_index: 0,
            shape_index: 0,
        }
    }

    /// Builds a [`BVHNode`] recursively using SAH partitioning.
    /// Returns the index of the new node in the nodes vector.
    ///
    /// [`BVHNode`]: enum.BVHNode.html
    ///
    pub fn build<T: BHShape>(
        shapes: &mut [T],
        indices: &mut [usize],
        nodes: &mut [BVHNode],
        parent_index: usize,
        depth: u32,
        node_index: usize,
        aabb_bounds: AABB,
        centroid_bounds: AABB,
    ) -> usize {
        // If there is only one element left, don't split anymore
        if indices.len() == 1 {
            let shape_index = indices[0];
            nodes[0] = BVHNode::Leaf {
                parent_index,
                shape_index,
            };
            // Let the shape know the index of the node that represents it.
            shapes[shape_index].set_bh_node_index(node_index);
            return node_index;
        }
        let mut parallel_recurse = false;
        if indices.len() > 64 {
            parallel_recurse = true;
        }

        // From here on we handle the recursive case. This dummy is required, because the children
        // must know their parent, and it's easier to update one parent node than the child nodes.
        nodes[0] = BVHNode::create_dummy();

        // Find the axis along which the shapes are spread the most.
        let split_axis = centroid_bounds.largest_axis();
        let split_axis_size = centroid_bounds.max[split_axis] - centroid_bounds.min[split_axis];

        // The following `if` partitions `indices` for recursively calling `BVH::build`.
        let (child_l_index, child_l_aabb, child_r_index, child_r_aabb) = if split_axis_size
            < EPSILON
        {
            // In this branch the shapes lie too close together so that splitting them in a
            // sensible way is not possible. Instead we just split the list of shapes in half.
            let (child_l_indices, child_r_indices) = indices.split_at_mut(indices.len() / 2);
            let (child_l_aabb, child_l_centroid) = joint_aabb_of_shapes(child_l_indices, shapes);
            let (child_r_aabb, child_r_centroid) = joint_aabb_of_shapes(child_r_indices, shapes);

            let next_nodes = &mut nodes[1..];
            let (l_nodes, r_nodes) = next_nodes.split_at_mut(child_l_indices.len() * 2 - 1);
            let child_l_index = node_index + 1;
            let child_r_index = node_index + 1 + l_nodes.len();
            // Proceed recursively.
            if parallel_recurse {
                // This is safe because shapes is only accessed using the indices and each index is unique
                let (shapes_a, shapes_b) = unsafe {
                    let ptr = shapes.as_mut_ptr();
                    let len = shapes.len();
                    let shapes_a = slice::from_raw_parts_mut(ptr, len);
                    let shapes_b = slice::from_raw_parts_mut(ptr, len);
                    (shapes_a, shapes_b)
                };
                rayon::join(
                    || {
                        BVHNode::build(
                            shapes_a,
                            child_l_indices,
                            l_nodes,
                            node_index,
                            depth + 1,
                            child_l_index,
                            child_l_aabb,
                            child_l_centroid,
                        )
                    },
                    || {
                        BVHNode::build(
                            shapes_b,
                            child_r_indices,
                            r_nodes,
                            node_index,
                            depth + 1,
                            child_r_index,
                            child_r_aabb,
                            child_r_centroid,
                        )
                    },
                );
            } else {
                BVHNode::build(
                    shapes,
                    child_l_indices,
                    l_nodes,
                    node_index,
                    depth + 1,
                    child_l_index,
                    child_l_aabb,
                    child_l_centroid,
                );
                BVHNode::build(
                    shapes,
                    child_r_indices,
                    r_nodes,
                    node_index,
                    depth + 1,
                    child_r_index,
                    child_r_aabb,
                    child_r_centroid,
                );
            }
            (child_l_index, child_l_aabb, child_r_index, child_r_aabb)
        } else {
            let (
                (child_l_aabb, child_l_centroid, child_l_indices),
                (child_r_aabb, child_r_centroid, child_r_indices),
            ) = BVHNode::build_buckets(
                shapes,
                indices,
                split_axis,
                split_axis_size,
                &centroid_bounds,
                &aabb_bounds,
            );

            let next_nodes = &mut nodes[1..];
            let (l_nodes, r_nodes) = next_nodes.split_at_mut(child_l_indices.len() * 2 - 1);

            let child_l_index = node_index + 1;
            let child_r_index = node_index + 1 + l_nodes.len();
            // Proceed recursively.

            if parallel_recurse {
                let (shapes_a, shapes_b) = unsafe {
                    let ptr = shapes.as_mut_ptr();
                    let len = shapes.len();
                    let shapes_a = slice::from_raw_parts_mut(ptr, len);
                    let shapes_b = slice::from_raw_parts_mut(ptr, len);
                    (shapes_a, shapes_b)
                };
                rayon::join(
                    || {
                        BVHNode::build(
                            shapes_a,
                            child_l_indices,
                            l_nodes,
                            node_index,
                            depth + 1,
                            child_l_index,
                            child_l_aabb,
                            child_l_centroid,
                        )
                    },
                    || {
                        BVHNode::build(
                            shapes_b,
                            child_r_indices,
                            r_nodes,
                            node_index,
                            depth + 1,
                            child_r_index,
                            child_r_aabb,
                            child_r_centroid,
                        )
                    },
                );
            } else {
                BVHNode::build(
                    shapes,
                    child_l_indices,
                    l_nodes,
                    node_index,
                    depth + 1,
                    child_l_index,
                    child_l_aabb,
                    child_l_centroid,
                );
                BVHNode::build(
                    shapes,
                    child_r_indices,
                    r_nodes,
                    node_index,
                    depth + 1,
                    child_r_index,
                    child_r_aabb,
                    child_r_centroid,
                );
            }
            (child_l_index, child_l_aabb, child_r_index, child_r_aabb)
        };

        // Construct the actual data structure and replace the dummy node.
        //assert!(!child_l_aabb.is_empty());
        //assert!(!child_r_aabb.is_empty());
        nodes[0] = BVHNode::Node {
            parent_index,
            child_l_aabb,
            child_l_index,
            child_r_aabb,
            child_r_index,
        };

        node_index
    }

    fn build_buckets<'a, T: BHShape>(
        shapes: &mut [T],
        indices: &'a mut [usize],
        split_axis: Axis,
        split_axis_size: Real,
        centroid_bounds: &AABB,
        aabb_bounds: &AABB,
    ) -> ((AABB, AABB, &'a mut [usize]), (AABB, AABB, &'a mut [usize])) {
        // Create six `Bucket`s, and six index assignment vector.
        // let mut buckets = [Bucket::empty(); NUM_BUCKETS];
        // let mut bucket_assignments: [SmallVec<[usize; 1024]>; NUM_BUCKETS] = Default::default();
        BUCKETS.with(move |buckets| {
            let bucket_assignments = &mut *buckets.borrow_mut();
            let mut buckets = [Bucket::empty(); NUM_BUCKETS];
            buckets.fill(Bucket::empty());
            for b in bucket_assignments.iter_mut() {
                b.clear();
            }

            // In this branch the `split_axis_size` is large enough to perform meaningful splits.
            // We start by assigning the shapes to `Bucket`s.
            for idx in indices.iter() {
                let shape = &shapes[*idx];
                let shape_aabb = shape.aabb();
                let shape_center = shape_aabb.center();

                // Get the relative position of the shape centroid `[0.0..1.0]`.
                let bucket_num_relative =
                    (shape_center[split_axis] - centroid_bounds.min[split_axis]) / split_axis_size;

                // Convert that to the actual `Bucket` number.
                let bucket_num = (bucket_num_relative * (NUM_BUCKETS as Real - 0.01)) as usize;

                // Extend the selected `Bucket` and add the index to the actual bucket.
                buckets[bucket_num].add_aabb(&shape_aabb);
                bucket_assignments[bucket_num].push(*idx);
            }

            // Compute the costs for each configuration and select the best configuration.
            let mut min_bucket = 0;
            let mut min_cost = Real::INFINITY;
            let mut child_l_aabb = AABB::empty();
            let mut child_l_centroid = AABB::empty();
            let mut child_r_aabb = AABB::empty();
            let mut child_r_centroid = AABB::empty();
            for i in 0..(NUM_BUCKETS - 1) {
                let (l_buckets, r_buckets) = buckets.split_at(i + 1);
                let child_l = l_buckets.iter().fold(Bucket::empty(), Bucket::join_bucket);
                let child_r = r_buckets.iter().fold(Bucket::empty(), Bucket::join_bucket);

                let cost = (child_l.size as Real * child_l.aabb.surface_area()
                    + child_r.size as Real * child_r.aabb.surface_area())
                    / aabb_bounds.surface_area();
                if cost < min_cost {
                    min_bucket = i;
                    min_cost = cost;
                    child_l_aabb = child_l.aabb;
                    child_l_centroid = child_l.centroid;
                    child_r_aabb = child_r.aabb;
                    child_r_centroid = child_r.centroid;
                }
            }
            // Join together all index buckets.
            // split input indices, loop over assignments and assign
            let (l_assignments, r_assignments) = bucket_assignments.split_at_mut(min_bucket + 1);

            let mut l_count = 0;
            for group in l_assignments.iter() {
                l_count += group.len();
            }

            let (child_l_indices, child_r_indices) = indices.split_at_mut(l_count);
            let mut i = 0;
            for group in l_assignments.iter() {
                for x in group {
                    child_l_indices[i] = *x;
                    i += 1;
                }
            }
            i = 0;
            for group in r_assignments.iter() {
                for x in group {
                    child_r_indices[i] = *x;
                    i += 1;
                }
            }

            (
                (child_l_aabb, child_l_centroid, child_l_indices),
                (child_r_aabb, child_r_centroid, child_r_indices),
            )
        })
    }

    /// Traverses the [`BVH`] recursively and returns all shapes whose [`AABB`] is
    /// intersected by the given [`Ray`].
    ///
    /// [`AABB`]: ../aabb/struct.AABB.html
    /// [`BVH`]: struct.BVH.html
    /// [`Ray`]: ../ray/struct.Ray.html
    ///
    pub fn traverse_recursive(
        nodes: &[BVHNode],
        node_index: usize,
        ray: &impl IntersectionAABB,
        indices: &mut Vec<usize>,
    ) {
        match nodes[node_index] {
            BVHNode::Node {
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
            BVHNode::Leaf { shape_index, .. } => {
                indices.push(shape_index);
            }
        }
    }
}

/// The [`BVH`] data structure. Contains the list of [`BVHNode`]s.
///
/// [`BVH`]: struct.BVH.html
///
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
#[cfg_attr(feature = "serde_impls", derive(serde::Serialize, serde::Deserialize))]
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
        let mut indices = (0..shapes.len()).collect::<Vec<usize>>();
        let expected_node_count = shapes.len() * 2 - 1;
        let mut nodes = Vec::with_capacity(expected_node_count);
        unsafe {
            nodes.set_len(expected_node_count);
        }
        //println!("shapes={} nodes={}", shapes.len(), nodes.len());
        let n = nodes.as_mut_slice();

        let (aabb, centroid) = joint_aabb_of_shapes(&indices, shapes);
        BVHNode::build(shapes, &mut indices, n, 0, 0, 0, aabb, centroid);
        BVH { nodes }
    }

    pub fn rebuild<Shape: BHShape>(&mut self, shapes: &mut [Shape]) {
        let mut indices = (0..shapes.len()).collect::<Vec<usize>>();
        let expected_node_count = shapes.len() * 2 - 1;
        let additional_nodes = self.nodes.capacity() as i32 - expected_node_count as i32;
        if additional_nodes > 0 {
            self.nodes.reserve(additional_nodes as usize);
        }
        unsafe {
            self.nodes.set_len(expected_node_count);
        }
        let n = self.nodes.as_mut_slice();
        let (aabb, centroid) = joint_aabb_of_shapes(&indices, shapes);
        BVHNode::build(shapes, &mut indices, n, 0, 0, 0, aabb, centroid);
    }

    /// Traverses the [`BVH`].
    /// Returns a subset of `shapes`, in which the [`AABB`]s of the elements were hit by `ray`.
    ///
    /// [`BVH`]: struct.BVH.html
    /// [`AABB`]: ../aabb/struct.AABB.html
    ///
    pub fn traverse<'a, Shape: Bounded>(
        &'a self,
        ray: &impl IntersectionAABB,
        shapes: &'a [Shape],
    ) -> Vec<&Shape> {
        let mut indices = Vec::new();
        BVHNode::traverse_recursive(&self.nodes, 0, ray, &mut indices);
        indices
            .iter()
            .map(|index| &shapes[*index])
            .collect::<Vec<_>>()
    }

    /// Creates a [`BVHTraverseIterator`] to traverse the [`BVH`].
    /// Returns a subset of `shapes`, in which the [`AABB`]s of the elements were hit by `ray`.
    ///
    /// [`BVH`]: struct.BVH.html
    /// [`AABB`]: ../aabb/struct.AABB.html
    ///
    pub fn traverse_iterator<'a, Shape: Bounded>(
        &'a self,
        test: &'a impl IntersectionAABB,
        shapes: &'a [Shape],
    ) -> BVHTraverseIterator<Shape> {
        BVHTraverseIterator::new(self, test, shapes)
    }

    /// Prints the [`BVH`] in a tree-like visualization.
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    pub fn pretty_print(&self) {
        self.print_node(0);
    }

    pub fn print_node(&self, node_index: usize) {
        let nodes = &self.nodes;
        match nodes[node_index] {
            BVHNode::Node {
                child_l_index,
                child_r_index,
                child_l_aabb,
                child_r_aabb,
                ..
            } => {
                let depth = nodes[node_index].depth(nodes);
                let padding: String = repeat(" ").take(depth as usize).collect();
                println!(
                    "{}node={} parent={}",
                    padding,
                    node_index,
                    nodes[node_index].parent()
                );
                println!("{}{} child_l {}", padding, child_l_index, child_l_aabb);
                self.print_node(child_l_index);
                println!("{}{} child_r {}", padding, child_r_index, child_r_aabb);
                self.print_node(child_r_index);
            }
            BVHNode::Leaf { shape_index, .. } => {
                let depth = nodes[node_index].depth(nodes);
                let padding: String = repeat(" ").take(depth as usize).collect();
                println!(
                    "{}node={} parent={}",
                    padding,
                    node_index,
                    nodes[node_index].parent()
                );
                println!("{}shape\t{:?}", padding, shape_index);
            }
        }
    }

    /// Verifies that the node at index `node_index` lies inside `expected_outer_aabb`,
    /// its parent index is equal to `expected_parent_index`, its depth is equal to
    /// `expected_depth`. Increares `node_count` by the number of visited nodes.
    fn is_consistent_subtree<Shape: BHShape>(
        &self,
        node_index: usize,
        expected_parent_index: usize,
        expected_outer_aabb: &AABB,
        expected_depth: u32,
        node_count: &mut usize,
        shapes: &[Shape],
    ) -> bool {
        *node_count += 1;
        match self.nodes[node_index] {
            BVHNode::Node {
                parent_index,
                child_l_index,
                child_l_aabb,
                child_r_index,
                child_r_aabb,
            } => {
                let depth = self.nodes[node_index].depth(self.nodes.as_slice());
                let correct_parent_index = expected_parent_index == parent_index;
                let correct_depth = expected_depth == depth;
                let left_aabb_in_parent =
                    expected_outer_aabb.approx_contains_aabb_eps(&child_l_aabb, EPSILON);
                let right_aabb_in_parent =
                    expected_outer_aabb.approx_contains_aabb_eps(&child_r_aabb, EPSILON);
                let left_subtree_consistent = self.is_consistent_subtree(
                    child_l_index,
                    node_index,
                    &child_l_aabb,
                    expected_depth + 1,
                    node_count,
                    shapes,
                );
                let right_subtree_consistent = self.is_consistent_subtree(
                    child_r_index,
                    node_index,
                    &child_r_aabb,
                    expected_depth + 1,
                    node_count,
                    shapes,
                );

                correct_parent_index
                    && correct_depth
                    && left_aabb_in_parent
                    && right_aabb_in_parent
                    && left_subtree_consistent
                    && right_subtree_consistent
            }
            BVHNode::Leaf {
                parent_index,
                shape_index,
            } => {
                let depth = self.nodes[node_index].depth(self.nodes.as_slice());
                let correct_parent_index = expected_parent_index == parent_index;
                let correct_depth = expected_depth == depth;
                let shape_aabb = shapes[shape_index].aabb();
                let shape_aabb_in_parent =
                    expected_outer_aabb.approx_contains_aabb_eps(&shape_aabb, EPSILON);

                correct_parent_index && correct_depth && shape_aabb_in_parent
            }
        }
    }

    /// Checks if all children of a node have the correct parent index, and that there is no
    /// detached subtree. Also checks if the `AABB` hierarchy is consistent.
    pub fn is_consistent<Shape: BHShape>(&self, shapes: &[Shape]) -> bool {
        // The root node of the bvh is not bounded by anything.
        let space = AABB {
            min: Point3::new(Real::NEG_INFINITY, Real::NEG_INFINITY, Real::NEG_INFINITY),
            max: Point3::new(Real::INFINITY, Real::INFINITY, Real::INFINITY),
        };

        // The counter for all nodes.
        let mut node_count = 0;
        let subtree_consistent =
            self.is_consistent_subtree(0, 0, &space, 0, &mut node_count, shapes);

        // Check if all nodes have been counted from the root node.
        // If this is false, it means we have a detached subtree.
        let is_connected = node_count == self.nodes.len();
        subtree_consistent && is_connected
    }

    /// Assert version of `is_consistent_subtree`.
    fn assert_consistent_subtree<Shape: BHShape>(
        &self,
        node_index: usize,
        expected_parent_index: usize,
        expected_outer_aabb: &AABB,
        expected_depth: u32,
        node_count: &mut usize,
        shapes: &[Shape],
    ) {
        if self.nodes.len() == 0 {
            return;
        }

        *node_count += 1;
        let node = &self.nodes[node_index];

        let parent = node.parent();
        assert_eq!(
            expected_parent_index, parent,
            "Wrong parent index. Expected: {}; Actual: {}",
            expected_parent_index, parent
        );
        let depth = node.depth(&self.nodes);
        assert_eq!(
            expected_depth, depth,
            "Wrong depth. Expected: {}; Actual: {}",
            expected_depth, depth
        );

        match *node {
            BVHNode::Node {
                child_l_index,
                child_l_aabb,
                child_r_index,
                child_r_aabb,
                ..
            } => {
                assert!(
                    expected_outer_aabb.approx_contains_aabb_eps(&child_l_aabb, EPSILON),
                    "Left child {} lies outside the expected bounds.
                         \tDepth: {}
                         \tBounds: {}
                         \tLeft child: {}
                         \tNode: {}",
                    child_l_index,
                    depth,
                    expected_outer_aabb,
                    child_l_aabb,
                    node_index
                );
                assert!(
                    expected_outer_aabb.approx_contains_aabb_eps(&child_r_aabb, EPSILON),
                    "Right child {} lies outside the expected bounds.
                         \tDepth: {}
                         \tBounds: {}
                         \tRight child: {}
                         \tNode: {}",
                    child_r_index,
                    depth,
                    expected_outer_aabb,
                    child_r_aabb,
                    node_index
                );
                self.assert_consistent_subtree(
                    child_l_index,
                    node_index,
                    &child_l_aabb,
                    expected_depth + 1,
                    node_count,
                    shapes,
                );
                self.assert_consistent_subtree(
                    child_r_index,
                    node_index,
                    &child_r_aabb,
                    expected_depth + 1,
                    node_count,
                    shapes,
                );
            }
            BVHNode::Leaf {
                shape_index,
                parent_index,
                ..
            } => {
                let shape_aabb = shapes[shape_index].aabb();
                assert!(
                    if parent != 0 {
                        expected_outer_aabb.relative_eq(&shape_aabb, EPSILON)
                    } else {
                        true
                    },
                    "Shape's AABB lies outside the expected bounds.\n\tBounds: {}\n\tShape: {}\n\tParent: {}",
                    expected_outer_aabb,
                    shape_aabb,
                    parent_index
                );
            }
        }
    }

    /// Assert version of `is_consistent`.
    pub fn assert_consistent<Shape: BHShape>(&self, shapes: &[Shape]) {
        // The root node of the bvh is not bounded by anything.
        let space = AABB {
            min: Point3::new(Real::NEG_INFINITY, Real::NEG_INFINITY, Real::NEG_INFINITY),
            max: Point3::new(Real::INFINITY, Real::INFINITY, Real::INFINITY),
        };

        // The counter for all nodes.
        let mut node_count = 0;
        self.assert_consistent_subtree(0, 0, &space, 0, &mut node_count, shapes);

        // Check if all nodes have been counted from the root node.
        // If this is false, it means we have a detached subtree.
        if node_count != self.nodes.len() {
            for x in node_count..self.nodes.len() {
                let node = self.nodes[x];
                match node {
                    BVHNode::Node {
                        parent_index,
                        child_l_index,
                        child_l_aabb,
                        child_r_index,
                        child_r_aabb,
                    } => {
                        println!(
                            "{}: parent_index={} child_l {} {} child_r {} {}",
                            x,
                            parent_index,
                            child_l_index,
                            child_l_aabb,
                            child_r_index,
                            child_r_aabb
                        );
                    }
                    BVHNode::Leaf {
                        parent_index,
                        shape_index,
                    } => {
                        println!("{}: parent={} shape={}", x, parent_index, shape_index);
                    }
                }
            }
        }
        assert_eq!(node_count, self.nodes.len(), "Detached subtree");
    }

    /// Check that the `AABB`s in the `BVH` are tight, which means, that parent `AABB`s are not
    /// larger than they should be. This function checks, whether the children of node `node_index`
    /// lie inside `outer_aabb`.
    pub fn assert_tight_subtree<Shape: BHShape>(
        &self,
        node_index: usize,
        outer_aabb: &AABB,
        shapes: &[Shape],
    ) {
        if let BVHNode::Node {
            child_l_index,
            child_l_aabb,
            child_r_index,
            child_r_aabb,
            ..
        } = self.nodes[node_index]
        {
            let joint_aabb = child_l_aabb.join(&child_r_aabb);
            if !joint_aabb.relative_eq(outer_aabb, EPSILON) {
                for _i in 0..shapes.len() {
                    //println!("s#{} {}", i, shapes[i].aabb())
                }
                //self.pretty_print();
                println!(
                    "{} real_aabb={} stored_aabb={}",
                    node_index, joint_aabb, outer_aabb
                );
            }
            assert!(joint_aabb.relative_eq(outer_aabb, EPSILON));
            self.assert_tight_subtree(child_l_index, &child_l_aabb, shapes);
            self.assert_tight_subtree(child_r_index, &child_r_aabb, shapes);
        }
    }

    /// Check that the `AABB`s in the `BVH` are tight, which means, that parent `AABB`s are not
    /// larger than they should be.
    pub fn assert_tight<Shape: BHShape>(&self, shapes: &[Shape]) {
        // When starting to check whether the `BVH` is tight, we cannot provide a minimum
        // outer `AABB`, therefore we compute the correct one in this instance.
        if let BVHNode::Node {
            child_l_aabb,
            child_r_aabb,
            ..
        } = self.nodes[0]
        {
            let joint_aabb = child_l_aabb.join(&child_r_aabb);
            self.assert_tight_subtree(0, &joint_aabb, shapes);
        }
    }

    /// Check that the `AABB`s in the `BVH` are tight, which means, that parent `AABB`s are not
    /// larger than they should be.
    pub fn assert_reachable<Shape: BHShape>(&self, shapes: &[Shape]) {
        for shape in shapes {
            let mut hit = false;
            for s in self.traverse_iterator(&shape.aabb(), shapes) {
                if s.bh_node_index() == shape.bh_node_index() {
                    hit = true;
                }
            }
            assert!(hit);
        }
    }
}

impl BoundingHierarchy for BVH {
    fn build<Shape: BHShape>(shapes: &mut [Shape]) -> BVH {
        BVH::build(shapes)
    }

    fn traverse<'a, Shape: Bounded>(
        &'a self,
        ray: &impl IntersectionAABB,
        shapes: &'a [Shape],
    ) -> Vec<&Shape> {
        self.traverse(ray, shapes)
    }

    fn pretty_print(&self) {
        self.pretty_print();
    }
}

#[cfg(test)]
mod tests {
    use crate::aabb::AABB;
    use crate::bounding_hierarchy::BHShape;
    use crate::bvh::{BVHNode, BVH};
    use crate::ray::Ray;
    use crate::testbase::{build_some_bh, traverse_some_bh, UnitBox};
    use crate::{Point3, Real, Vector3};
    use itertools::Itertools;

    #[test]
    /// Tests whether the building procedure succeeds in not failing.
    fn test_build_bvh() {
        let (shapes, bvh) = build_some_bh::<BVH>();
        bvh.is_consistent(shapes.as_slice());
        bvh.pretty_print();
    }

    #[test]
    /// Runs some primitive tests for intersections of a ray with a fixed scene given as a BVH.
    fn test_traverse_bvh() {
        traverse_some_bh::<BVH>();
    }

    #[test]
    fn test_add_bvh() {
        let mut shapes = Vec::new();

        for x in -1..2 {
            shapes.push(UnitBox::new(x, Point3::new(x as Real, 0.0, 0.0)));
        }

        let mut bvh = BVH::build(&mut shapes);
        bvh.pretty_print();

        let test = AABB::empty()
            .grow(&Point3::new(1.6, 0.0, 0.0))
            .grow(&Point3::new(2.4, 1.0, 1.0));

        let res = bvh.traverse(&test, &shapes);
        assert_eq!(res.len(), 0);

        let x = 2;
        shapes.push(UnitBox::new(x, Point3::new(x as Real, 0.0, 0.0)));
        let len = shapes.len() - 1;
        bvh.add_node(&mut shapes, len);

        bvh.pretty_print();
        bvh.rebuild(&mut shapes);
        let res = bvh.traverse(&test, &shapes);
        assert_eq!(res.len(), 1);

        let x = 50;
        shapes.push(UnitBox::new(x, Point3::new(x as Real, 0.0, 0.0)));
        let len = shapes.len() - 1;
        bvh.add_node(&mut shapes, len);
        bvh.pretty_print();
        let res = bvh.traverse(&test, &shapes);
        assert_eq!(res.len(), 1);

        let test = AABB::empty()
            .grow(&Point3::new(49.6, 0.0, 0.0))
            .grow(&Point3::new(52.4, 1.0, 1.0));
        let res = bvh.traverse(&test, &shapes);
        assert_eq!(res.len(), 1);
    }

    #[test]
    fn test_remove_bvh() {
        let mut shapes = Vec::new();
        for x in -1..1 {
            shapes.push(UnitBox::new(x, Point3::new(x as Real, 0.0, 0.0)));
        }

        let mut bvh = BVH::build(&mut shapes);

        bvh.pretty_print();
        println!("------");
        println!("{:?}", bvh.nodes.len());
        bvh.remove_node(&mut shapes, 0, true);

        println!("{:?}", bvh.nodes.len());
        println!("------");
        bvh.pretty_print();
    }

    #[test]
    fn test_accuracy_after_bvh_remove() {
        let mut shapes = Vec::new();
        for x in -25..25 {
            shapes.push(UnitBox::new(x, Point3::new(x as Real, 0.0, 0.0)));
        }

        let mut bvh = BVH::build(&mut shapes);
        bvh.pretty_print();
        bvh.assert_consistent(shapes.as_slice());

        fn test_x(bvh: &BVH, x: Real, count: usize, shapes: &[UnitBox]) {
            let dir = Vector3::new(0.0, -1.0, 0.0);

            let ray = Ray::new(Vector3::new(x, 2.0, 0.0), dir);
            let res = bvh.traverse(&ray, shapes);
            if count == 0 && res.len() > 0 {
                println!("hit={} node={}", res[0].pos, res[0].bh_node_index());
            }
            assert_eq!(res.len(), count);
        }

        test_x(&bvh, 2.0, 1, &shapes);

        for x in -23..23 {
            let point = Point3::new(x as Real, 0.0, 0.0);
            let mut delete_i = 0;
            for i in 0..shapes.len() {
                if shapes[i].pos.distance_squared(point) < 0.01 {
                    delete_i = i;
                    break;
                }
            }
            println!("Testing {}", x);
            bvh.pretty_print();
            println!("Ensuring x={} shape[{}] is present", x, delete_i);
            test_x(&bvh, x as Real, 1, &shapes);
            println!("Deleting x={} shape[{}]", x, delete_i);
            bvh.remove_node(&mut shapes, delete_i, true);
            shapes.truncate(shapes.len() - 1);
            bvh.pretty_print();
            println!("Ensuring {} [{}] is gone", x, delete_i);
            test_x(&bvh, x as Real, 0, &shapes);
        }
    }

    #[test]
    fn test_random_deletions() {
        let xs = -3..3;
        let x_values = xs.clone().collect::<Vec<i32>>();
        for x_values in xs.clone().permutations(x_values.len() - 1) {
            let mut shapes = Vec::new();
            for x in xs.clone() {
                shapes.push(UnitBox::new(x, Point3::new(x as Real, 0.0, 0.0)));
            }
            let mut bvh = BVH::build(&mut shapes);

            //bvh.pretty_print();
            for x_i in 0..x_values.len() {
                let x = x_values[x_i];

                let point = Point3::new(x as Real, 0.0, 0.0);
                let mut delete_i = 0;
                for i in 0..shapes.len() {
                    if shapes[i].pos.distance_squared(point) < 0.01 {
                        delete_i = i;
                        break;
                    }
                }
                //dbg!(delete_i, shapes.len());
                bvh.remove_node(&mut shapes, delete_i, true);
                shapes.truncate(shapes.len() - 1);
                assert_eq!(shapes.len(), x_values.len() - x_i);
                //println!("--------------------------");
                //bvh.pretty_print();
                bvh.assert_consistent(shapes.as_slice());
                bvh.assert_tight(shapes.as_slice());
            }
        }
    }

    #[test]
    fn test_add_consistency() {
        let mut shapes = Vec::new();
        for x in -25..25 {
            shapes.push(UnitBox::new(x, Point3::new(x as Real, 0.0, 0.0)));
        }

        let (left, right) = shapes.split_at_mut(10);

        let mut bvh = BVH::build(left);
        bvh.pretty_print();
        bvh.assert_consistent(left);

        for i in 0..right.len() {
            let x = i + 10;
            bvh.add_node(&mut shapes, x);
            bvh.assert_tight(shapes.as_slice());
            bvh.assert_consistent(shapes.as_slice());
        }
    }

    #[test]
    /// Verify contents of the bounding hierarchy for a fixed scene structure
    fn test_bvh_shape_indices() {
        use std::collections::HashSet;

        let (all_shapes, bh) = build_some_bh::<BVH>();

        // It should find all shape indices.
        let expected_shapes: HashSet<_> = (0..all_shapes.len()).collect();
        let mut found_shapes = HashSet::new();

        for node in bh.nodes.iter() {
            match *node {
                BVHNode::Node { .. } => {
                    assert_eq!(node.shape_index(), None);
                }
                BVHNode::Leaf { .. } => {
                    found_shapes.insert(
                        node.shape_index()
                            .expect("getting a shape index from a leaf node"),
                    );
                }
            }
        }

        assert_eq!(expected_shapes, found_shapes);
    }
}

#[cfg(all(feature = "bench", test))]
mod bench {
    use crate::bounding_hierarchy::BHShape;
    use crate::bvh::BVH;
    use crate::testbase::{
        build_1200_triangles_bh, build_120k_triangles_bh, build_12k_triangles_bh, create_n_cubes,
        default_bounds, intersect_1200_triangles_bh, intersect_120k_triangles_bh,
        intersect_12k_triangles_bh, intersect_bh, load_sponza_scene,
    };

    #[bench]
    /// Benchmark the construction of a `BVH` with 1,200 triangles.
    fn bench_build_1200_triangles_bvh(b: &mut ::test::Bencher) {
        build_1200_triangles_bh::<BVH>(b);
    }

    #[bench]
    /// Benchmark the construction of a `BVH` with 12,000 triangles.
    fn bench_build_12k_triangles_bvh(b: &mut ::test::Bencher) {
        build_12k_triangles_bh::<BVH>(b);
    }

    #[bench]
    /// Benchmark the construction of a `BVH` with 120,000 triangles.
    fn bench_build_120k_triangles_bvh(b: &mut ::test::Bencher) {
        build_120k_triangles_bh::<BVH>(b);
    }

    #[bench]
    /// Benchmark the construction of a `BVH` for the Sponza scene.
    fn bench_build_sponza_bvh(b: &mut ::test::Bencher) {
        let (mut triangles, _) = load_sponza_scene();
        b.iter(|| {
            BVH::build(&mut triangles);
        });
    }

    #[cfg(feature = "bench")]
    fn add_triangles_bvh(n: usize, b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(n, &bounds);
        b.iter(|| {
            build_by_add(triangles.as_mut_slice());
        });
    }

    #[cfg(feature = "bench")]
    fn build_by_add<T: BHShape>(shapes: &mut [T]) -> BVH {
        let (first, rest) = shapes.split_at_mut(1);
        let mut bvh = BVH::build(first);
        for i in 1..shapes.len() {
            bvh.add_node(shapes, i)
        }
        bvh
    }

    #[cfg(feature = "bench")]
    pub fn intersect_n_triangles_add(n: usize, b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(n, &bounds);
        let bh = build_by_add(&mut triangles);
        intersect_bh(&bh, &triangles, &bounds, b)
    }

    #[bench]
    /// Benchmark the construction of a `BVH` for the Sponza scene.
    fn build_1200_triangles_add(b: &mut ::test::Bencher) {
        add_triangles_bvh(1200, b)
    }

    #[bench]
    /// Benchmark the construction of a `BVH` for the Sponza scene.
    fn build_12k_triangles_add(b: &mut ::test::Bencher) {
        add_triangles_bvh(12000, b)
    }

    /*
    #[bench]
    /// Benchmark the construction of a `BVH` for the Sponza scene.
    fn build_120k_triangles_add(b: &mut ::test::Bencher) {
        add_triangles_bvh(120000, b)
    }
    */

    #[bench]
    /// Benchmark intersecting 1,200 triangles using the recursive `BVH`.
    fn bench_intersect_1200_triangles_bvh(b: &mut ::test::Bencher) {
        intersect_1200_triangles_bh::<BVH>(b);
    }

    #[bench]
    /// Benchmark intersecting 12,000 triangles using the recursive `BVH`.
    fn bench_intersect_12k_triangles_bvh(b: &mut ::test::Bencher) {
        intersect_12k_triangles_bh::<BVH>(b);
    }

    #[bench]
    /// Benchmark intersecting 120,000 triangles using the recursive `BVH`.
    fn bench_intersect_120k_triangles_bvh(b: &mut ::test::Bencher) {
        intersect_120k_triangles_bh::<BVH>(b);
    }

    #[bench]
    /// Benchmark intersecting 1,200 triangles using the recursive `BVH`.
    fn bench_intersect_1200_triangles_bvh_add(mut b: &mut ::test::Bencher) {
        intersect_n_triangles_add(1200, &mut b);
    }

    #[bench]
    /// Benchmark intersecting 1,200 triangles using the recursive `BVH`.
    fn bench_intersect_12000_triangles_bvh_add(mut b: &mut ::test::Bencher) {
        intersect_n_triangles_add(12000, &mut b);
    }

    #[bench]
    /// Benchmark the traversal of a `BVH` with the Sponza scene.
    fn bench_intersect_sponza_bvh(b: &mut ::test::Bencher) {
        let (mut triangles, bounds) = load_sponza_scene();
        let bvh = BVH::build(&mut triangles);
        intersect_bh(&bvh, &triangles, &bounds, b)
    }
}
