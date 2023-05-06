//! This module defines [`Bvh`] and [`BvhNode`] and functions for building and traversing it.
//!
//! [`Bvh`]: struct.Bvh.html
//! [`BvhNode`]: struct.BvhNode.html
//!

use nalgebra::{ClosedAdd, ClosedMul, ClosedSub, Scalar, SimdPartialOrd};
use num::{Float, FromPrimitive, Signed, ToPrimitive, Zero};

use crate::aabb::{Aabb, Bounded};
use crate::bounding_hierarchy::{BHShape, BoundingHierarchy};
use crate::bvh::iter::BvhTraverseIterator;
use crate::ray::Ray;
use crate::utils::{concatenate_vectors, joint_aabb_of_shapes, Bucket};

/// The [`BvhNode`] enum that describes a node in a [`Bvh`].
/// It's either a leaf node and references a shape (by holding its index)
/// or a regular node that has two child nodes.
/// The non-leaf node stores the [`Aabb`]s of its children.
///
/// [`Aabb`]: ../aabb/struct.Aabb.html
/// [`Bvh`]: struct.Bvh.html
/// [`Bvh`]: struct.BvhNode.html
///
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BvhNode<T: Scalar + Copy, const D: usize> {
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

        /// The convex hull of the shapes' [`Aabb`]'s in child_l.
        child_l_aabb: Aabb<T, D>,

        /// Index of the right subtree's root node.
        child_r_index: usize,

        /// The convex hull of the shapes' [`Aabb`]'s in child_r.
        child_r_aabb: Aabb<T, D>,
    },
}

impl<T: Scalar + Copy, const D: usize> PartialEq for BvhNode<T, D> {
    // TODO Consider also comparing [`Aabbs`]
    fn eq(&self, other: &BvhNode<T, D>) -> bool {
        match (self, other) {
            (
                &BvhNode::Node {
                    parent_index: self_parent_index,
                    child_l_index: self_child_l_index,
                    child_r_index: self_child_r_index,
                    ..
                },
                &BvhNode::Node {
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
                &BvhNode::Leaf {
                    parent_index: self_parent_index,
                    shape_index: self_shape_index,
                },
                &BvhNode::Leaf {
                    parent_index: other_parent_index,
                    shape_index: other_shape_index,
                },
            ) => {
                self_parent_index == other_parent_index
                    && self_shape_index == other_shape_index
            }
            _ => false,
        }
    }
}

impl<T: Scalar + Copy, const D: usize> BvhNode<T, D> {
    /// Returns the index of the parent node.
    pub fn parent(&self) -> usize {
        match *self {
            BvhNode::Node { parent_index, .. } | BvhNode::Leaf { parent_index, .. } => parent_index,
        }
    }

    /// Returns a mutable reference to the parent node index.
    pub fn parent_mut(&mut self) -> &mut usize {
        match *self {
            BvhNode::Node {
                ref mut parent_index,
                ..
            }
            | BvhNode::Leaf {
                ref mut parent_index,
                ..
            } => parent_index,
        }
    }

    /// Returns the index of the left child node.
    pub fn child_l(&self) -> usize {
        match *self {
            BvhNode::Node { child_l_index, .. } => child_l_index,
            _ => panic!("Tried to get the left child of a leaf node."),
        }
    }

    /// Returns the index of the left child node.
    pub fn child_l_mut(&mut self) -> &mut usize {
        match *self {
            BvhNode::Node {
                ref mut child_l_index,
                ..
            } => child_l_index,
            _ => panic!("Tried to get the left child of a leaf node."),
        }
    }

    /// Returns the `Aabb` of the right child node.
    pub fn child_l_aabb(&self) -> Aabb<T, D> {
        match *self {
            BvhNode::Node { child_l_aabb, .. } => child_l_aabb,
            _ => panic!(),
        }
    }

    /// Returns a mutable reference to the [`Aabb`] of the left child node.
    pub fn child_l_aabb_mut(&mut self) -> &mut Aabb<T, D> {
        match *self {
            BvhNode::Node {
                ref mut child_l_aabb,
                ..
            } => child_l_aabb,
            _ => panic!("Tried to get the left child's `Aabb` of a leaf node."),
        }
    }

    /// Returns the index of the right child node.
    pub fn child_r(&self) -> usize {
        match *self {
            BvhNode::Node { child_r_index, .. } => child_r_index,
            _ => panic!("Tried to get the right child of a leaf node."),
        }
    }
    
    /// Returns the index of the right child node.
    pub fn child_r_mut(&mut self) -> &mut usize {
        match *self {
            BvhNode::Node {
                ref mut child_r_index,
                ..
            } => child_r_index,
            _ => panic!("Tried to get the right child of a leaf node."),
        }
    }

    /// Returns the [`Aabb`] of the right child node.
    pub fn child_r_aabb(&self) -> Aabb<T, D> {
        match *self {
            BvhNode::Node { child_r_aabb, .. } => child_r_aabb,
            _ => panic!(),
        }
    }

    /// Returns a mutable reference to the [`Aabb`] of the right child node.
    pub fn child_r_aabb_mut(&mut self) -> &mut Aabb<T, D> {
        match *self {
            BvhNode::Node {
                ref mut child_r_aabb,
                ..
            } => child_r_aabb,
            _ => panic!("Tried to get the right child's `Aabb` of a leaf node."),
        }
    }

    /// Gets the [`Aabb`] for a [`BvhNode`].
    /// Returns the shape's [`Aabb`] for leaves, and the joined [`Aabb`] of
    /// the two children's [`Aabb`]'s for non-leaves.
    pub fn get_node_aabb<Shape: BHShape<T, D>>(&self, shapes: &[Shape]) -> Aabb<T, D>
    where
        T: SimdPartialOrd,
    {
        match *self {
            BvhNode::Node {
                child_l_aabb,
                child_r_aabb,
                ..
            } => child_l_aabb.join(&child_r_aabb),
            BvhNode::Leaf { shape_index, .. } => shapes[shape_index].aabb(),
        }
    }

    /// Returns the index of the shape contained within the node if is a leaf,
    /// or [`None`] if it is an interior node.
    pub fn shape_index(&self) -> Option<usize> {
        match *self {
            BvhNode::Leaf { shape_index, .. } => Some(shape_index),
            _ => None,
        }
    }

    /// Returns the index of the shape contained within the node if is a leaf,
    /// or `None` if it is an interior node.
    pub fn shape_index_mut(&mut self) -> Option<&mut usize> {
        match *self {
            BvhNode::Leaf {
                ref mut shape_index,
                ..
            } => Some(shape_index),
            _ => None,
        }
    }

    /// The build function sometimes needs to add nodes while their data is not available yet.
    /// A dummy cerated by this function serves the purpose of being changed later on.
    fn create_dummy() -> BvhNode<T, D> {
        BvhNode::Leaf {
            parent_index: 0,
            shape_index: 0,
        }
    }

    /// Builds a [`BvhNode`] recursively using SAH partitioning.
    /// Returns the index of the new node in the nodes vector.
    ///
    /// [`BvhNode`]: enum.BvhNode.html
    ///
    pub fn build<S: BHShape<T, D>>(
        shapes: &mut [S],
        indices: &[usize],
        nodes: &mut Vec<BvhNode<T, D>>,
        parent_index: usize,
        depth: u32,
    ) -> usize
    where
        T: FromPrimitive + ClosedSub + ClosedAdd + SimdPartialOrd + ClosedMul + Float,
    {
        // Helper function to accumulate the Aabb joint and the centroids Aabb
        fn grow_convex_hull<
            T: Scalar
                + Copy
                + ClosedSub
                + ClosedMul
                + SimdPartialOrd
                + ClosedAdd
                + FromPrimitive
                + ToPrimitive,
            const D: usize,
        >(
            convex_hull: (Aabb<T, D>, Aabb<T, D>),
            shape_aabb: &Aabb<T, D>,
        ) -> (Aabb<T, D>, Aabb<T, D>) {
            let center = &shape_aabb.center();
            let convex_hull_aabbs = &convex_hull.0;
            let convex_hull_centroids = &convex_hull.1;
            (
                convex_hull_aabbs.join(shape_aabb),
                convex_hull_centroids.grow(center),
            )
        }

        let mut convex_hull = Default::default();
        for index in indices {
            convex_hull = grow_convex_hull(convex_hull, &shapes[*index].aabb());
        }
        let (aabb_bounds, centroid_bounds) = convex_hull;

        // If there is only one element left, don't split anymore
        if indices.len() == 1 {
            let shape_index = indices[0];
            let node_index = nodes.len();
            nodes.push(BvhNode::Leaf {
                parent_index,
                shape_index,
            });
            // Let the shape know the index of the node that represents it.
            shapes[shape_index].set_bh_node_index(node_index);
            return node_index;
        }

        // From here on we handle the recursive case. This dummy is required, because the children
        // must know their parent, and it's easier to update one parent node than the child nodes.
        let node_index = nodes.len();
        nodes.push(BvhNode::create_dummy());

        // Find the axis along which the shapes are spread the most.
        let split_axis = centroid_bounds.largest_axis();
        let split_axis_size = centroid_bounds.max[split_axis] - centroid_bounds.min[split_axis];

        // The following `if` partitions `indices` for recursively calling `Bvh::build`.
        let (child_l_index, child_l_aabb, child_r_index, child_r_aabb) = if split_axis_size
            < T::epsilon()
        {
            // In this branch the shapes lie too close together so that splitting them in a
            // sensible way is not possible. Instead we just split the list of shapes in half.
            let (child_l_indices, child_r_indices) = indices.split_at(indices.len() / 2);
            let child_l_aabb = joint_aabb_of_shapes(child_l_indices, shapes);
            let child_r_aabb = joint_aabb_of_shapes(child_r_indices, shapes);

            // Proceed recursively.
            let child_l_index =
                BvhNode::build(shapes, child_l_indices, nodes, node_index, depth + 1);
            let child_r_index =
                BvhNode::build(shapes, child_r_indices, nodes, node_index, depth + 1);
            (child_l_index, child_l_aabb, child_r_index, child_r_aabb)
        } else {
            // Create six `Bucket`s, and six index assignment vector.
            const NUM_BUCKETS: usize = 6;
            let mut buckets = [Bucket::empty(); NUM_BUCKETS];
            let mut bucket_assignments: [Vec<usize>; NUM_BUCKETS] = Default::default();

            // In this branch the `split_axis_size` is large enough to perform meaningful splits.
            // We start by assigning the shapes to `Bucket`s.
            for idx in indices {
                let shape = &shapes[*idx];
                let shape_aabb = shape.aabb();
                let shape_center = shape_aabb.center();

                // Get the relative position of the shape centroid `[0.0..1.0]`.
                let bucket_num_relative =
                    (shape_center[split_axis] - centroid_bounds.min[split_axis]) / split_axis_size;

                // Convert that to the actual `Bucket` number.
                // TODO check this for integers
                let bucket_num = (bucket_num_relative
                    * (T::from_usize(NUM_BUCKETS).unwrap() - T::from_f32(0.01).unwrap()))
                .to_usize()
                .unwrap();

                // Extend the selected `Bucket` and add the index to the actual bucket.
                buckets[bucket_num].add_aabb(&shape_aabb);
                bucket_assignments[bucket_num].push(*idx);
            }

            // Compute the costs for each configuration and select the best configuration.
            let mut min_bucket = 0;
            let mut min_cost = T::infinity();
            let mut child_l_aabb = Aabb::empty();
            let mut child_r_aabb = Aabb::empty();
            for i in 0..(NUM_BUCKETS - 1) {
                let (l_buckets, r_buckets) = buckets.split_at(i + 1);
                let child_l = l_buckets.iter().fold(Bucket::empty(), Bucket::join_bucket);
                let child_r = r_buckets.iter().fold(Bucket::empty(), Bucket::join_bucket);

                let cost = (T::from_usize(child_l.size).unwrap() * child_l.aabb.surface_area()
                    + T::from_usize(child_r.size).unwrap() * child_r.aabb.surface_area())
                    / aabb_bounds.surface_area();
                if cost < min_cost {
                    min_bucket = i;
                    min_cost = cost;
                    child_l_aabb = child_l.aabb;
                    child_r_aabb = child_r.aabb;
                }
            }

            // Join together all index buckets.
            let (l_assignments, r_assignments) = bucket_assignments.split_at_mut(min_bucket + 1);
            let child_l_indices = concatenate_vectors(l_assignments);
            let child_r_indices = concatenate_vectors(r_assignments);

            // Proceed recursively.
            let child_l_index =
                BvhNode::build(shapes, &child_l_indices, nodes, node_index, depth + 1);
            let child_r_index =
                BvhNode::build(shapes, &child_r_indices, nodes, node_index, depth + 1);
            (child_l_index, child_l_aabb, child_r_index, child_r_aabb)
        };

        // Construct the actual data structure and replace the dummy node.
        assert!(!child_l_aabb.is_empty());
        assert!(!child_r_aabb.is_empty());
        nodes[node_index] = BvhNode::Node {
            parent_index,
            child_l_aabb,
            child_l_index,
            child_r_aabb,
            child_r_index,
        };

        node_index
    }

    /// Traverses the [`Bvh`] recursively and returns all shapes whose [`Aabb`] is
    /// intersected by the given [`Ray`].
    ///
    /// [`Aabb`]: ../aabb/struct.Aabb.html
    /// [`Bvh`]: struct.Bvh.html
    /// [`Ray`]: ../ray/struct.Ray.html
    ///
    pub fn traverse_recursive(
        nodes: &[BvhNode<T, D>],
        node_index: usize,
        ray: &Ray<T, D>,
        indices: &mut Vec<usize>,
    ) where
        T: PartialOrd + Zero + ClosedSub + ClosedMul + SimdPartialOrd,
    {
        match nodes[node_index] {
            BvhNode::Node {
                ref child_l_aabb,
                child_l_index,
                ref child_r_aabb,
                child_r_index,
                ..
            } => {
                if ray.intersects_aabb(child_l_aabb) {
                    BvhNode::traverse_recursive(nodes, child_l_index, ray, indices);
                }
                if ray.intersects_aabb(child_r_aabb) {
                    BvhNode::traverse_recursive(nodes, child_r_index, ray, indices);
                }
            }
            BvhNode::Leaf { shape_index, .. } => {
                indices.push(shape_index);
            }
        }
    }
}

/// The [`Bvh`] data structure. Contains the list of [`BvhNode`]s.
///
/// [`Bvh`]: struct.Bvh.html
///
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Bvh<T: Scalar + Copy, const D: usize> {
    /// The list of nodes of the [`Bvh`].
    ///
    /// [`Bvh`]: struct.Bvh.html
    ///
    pub nodes: Vec<BvhNode<T, D>>,
}

impl<T: Scalar + Copy, const D: usize> Bvh<T, D> {
    /// Creates a new [`Bvh`] from the `shapes` slice.
    ///
    /// [`Bvh`]: struct.Bvh.html
    ///
    pub fn build<Shape: BHShape<T, D>>(shapes: &mut [Shape]) -> Bvh<T, D>
    where
        T: FromPrimitive + ToPrimitive + Float + ClosedSub + ClosedAdd + ClosedMul + SimdPartialOrd,
    {
        let indices = (0..shapes.len()).collect::<Vec<usize>>();
        let expected_node_count = shapes.len() * 2;
        let mut nodes = Vec::with_capacity(expected_node_count);
        BvhNode::build(shapes, &indices, &mut nodes, 0, 0);
        Bvh { nodes }
    }

    /// Traverses the [`Bvh`].
    /// Returns a subset of `shapes`, in which the [`Aabb`]s of the elements were hit by `ray`.
    ///
    /// [`Bvh`]: struct.Bvh.html
    /// [`Aabb`]: ../aabb/struct.Aabb.html
    ///
    pub fn traverse<'a, Shape: Bounded<T, D>>(
        &'a self,
        ray: &Ray<T, D>,
        shapes: &'a [Shape],
    ) -> Vec<&Shape>
    where
        T: PartialOrd + Zero + ClosedSub + ClosedMul + SimdPartialOrd,
    {
        let mut indices = Vec::new();
        BvhNode::traverse_recursive(&self.nodes, 0, ray, &mut indices);
        indices
            .iter()
            .map(|index| &shapes[*index])
            .collect::<Vec<_>>()
    }

    /// Creates a [`BvhTraverseIterator`] to traverse the [`Bvh`].
    /// Returns a subset of `shapes`, in which the [`Aabb`]s of the elements were hit by `ray`.
    ///
    /// [`Bvh`]: struct.Bvh.html
    /// [`Aabb`]: ../aabb/struct.Aabb.html
    ///
    pub fn traverse_iterator<'bvh, 'shape, Shape: Bounded<T, D>>(
        &'bvh self,
        ray: &'bvh Ray<T, D>,
        shapes: &'shape [Shape],
    ) -> BvhTraverseIterator<'bvh, 'shape, T, D, Shape>
    where
        T: SimdPartialOrd + ClosedSub + PartialOrd + ClosedMul + Zero,
    {
        BvhTraverseIterator::new(self, ray, shapes)
    }

    /// Prints the [`Bvh`] in a tree-like visualization.
    ///
    /// [`Bvh`]: struct.Bvh.html
    ///
    pub fn pretty_print(&self)
    where
        T: std::fmt::Display,
    {
        let nodes = &self.nodes;
        fn print_node<T: Scalar + Copy, const D: usize>(nodes: &[BvhNode<T, D>], node_index: usize)
        where
            T: std::fmt::Display,
        {
            match nodes[node_index] {
                BvhNode::Node {
                    child_l_index,
                    child_r_index,
                    child_l_aabb,
                    child_r_aabb,
                    ..
                } => {
                    let padding: String = " ".repeat(0 as usize); // TODO:(FIX)
                    println!("{}child_l {}", padding, child_l_aabb);
                    print_node(nodes, child_l_index);
                    println!("{}child_r {}", padding, child_r_aabb);
                    print_node(nodes, child_r_index);
                }
                BvhNode::Leaf {
                    shape_index, ..
                } => {
                    let padding: String = " ".repeat(0 as usize); // TODO:(FIX)
                    println!("{}shape\t{:?}", padding, shape_index);
                }
            }
        }
        print_node(nodes, 0);
    }

    /// Verifies that the node at index `node_index` lies inside `expected_outer_aabb`,
    /// its parent index is equal to `expected_parent_index`, its depth is equal to
    /// `expected_depth`. Increares `node_count` by the number of visited nodes.
    fn is_consistent_subtree<Shape: BHShape<T, D>>(
        &self,
        node_index: usize,
        expected_parent_index: usize,
        expected_outer_aabb: &Aabb<T, D>,
        expected_depth: u32,
        node_count: &mut usize,
        shapes: &[Shape],
    ) -> bool
    where
        T: Float + ClosedSub,
    {
        *node_count += 1;
        match self.nodes[node_index] {
            BvhNode::Node {
                parent_index,
                child_l_index,
                child_l_aabb,
                child_r_index,
                child_r_aabb,
            } => {
                let correct_parent_index = expected_parent_index == parent_index;
                let left_aabb_in_parent =
                    expected_outer_aabb.approx_contains_aabb_eps(&child_l_aabb, T::epsilon());
                let right_aabb_in_parent =
                    expected_outer_aabb.approx_contains_aabb_eps(&child_r_aabb, T::epsilon());
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
                    && left_aabb_in_parent
                    && right_aabb_in_parent
                    && left_subtree_consistent
                    && right_subtree_consistent
            }
            BvhNode::Leaf {
                parent_index,
                shape_index,
            } => {
                let correct_parent_index = expected_parent_index == parent_index;
                let shape_aabb = shapes[shape_index].aabb();
                let shape_aabb_in_parent =
                    expected_outer_aabb.approx_contains_aabb_eps(&shape_aabb, T::epsilon());

                correct_parent_index && shape_aabb_in_parent
            }
        }
    }

    /// Checks if all children of a node have the correct parent index, and that there is no
    /// detached subtree. Also checks if the `Aabb` hierarchy is consistent.
    pub fn is_consistent<Shape: BHShape<T, D>>(&self, shapes: &[Shape]) -> bool
    where
        T: Float + ClosedSub,
    {
        // The root node of the bvh is not bounded by anything.
        let space = Aabb::infinite();

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
    fn assert_consistent_subtree<Shape: BHShape<T, D>>(
        &self,
        node_index: usize,
        expected_parent_index: usize,
        expected_outer_aabb: &Aabb<T, D>,
        expected_depth: u32,
        node_count: &mut usize,
        shapes: &[Shape],
    ) where
        T: Float + ClosedSub + std::fmt::Display,
    {
        *node_count += 1;
        let node = &self.nodes[node_index];

        let parent = node.parent();
        assert_eq!(
            expected_parent_index, parent,
            "Wrong parent index. Expected: {}; Actual: {}",
            expected_parent_index, parent
        );

        match *node {
            BvhNode::Node {
                child_l_index,
                child_l_aabb,
                child_r_index,
                child_r_aabb,
                ..
            } => {
                assert!(
                    expected_outer_aabb.approx_contains_aabb_eps(&child_l_aabb, T::epsilon()),
                    "Left child lies outside the expected bounds.
                         \tBounds: {}
                         \tLeft child: {}",
                    expected_outer_aabb,
                    child_l_aabb
                );
                assert!(
                    expected_outer_aabb.approx_contains_aabb_eps(&child_r_aabb, T::epsilon()),
                    "Right child lies outside the expected bounds.
                         \tBounds: {}
                         \tRight child: {}",
                    expected_outer_aabb,
                    child_r_aabb
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
            BvhNode::Leaf { shape_index, .. } => {
                let shape_aabb = shapes[shape_index].aabb();
                assert!(
                    expected_outer_aabb.approx_contains_aabb_eps(&shape_aabb, T::epsilon()),
                    "Shape's Aabb lies outside the expected bounds.\n\tBounds: {}\n\tShape: {}",
                    expected_outer_aabb,
                    shape_aabb
                );
            }
        }
    }

    /// Assert version of `is_consistent`.
    pub fn assert_consistent<Shape: BHShape<T, D>>(&self, shapes: &[Shape])
    where
        T: Float + ClosedSub + std::fmt::Display,
    {
        // The root node of the bvh is not bounded by anything.
        let space = Aabb::infinite();

        // The counter for all nodes.
        let mut node_count = 0;
        self.assert_consistent_subtree(0, 0, &space, 0, &mut node_count, shapes);

        // Check if all nodes have been counted from the root node.
        // If this is false, it means we have a detached subtree.
        assert_eq!(node_count, self.nodes.len(), "Detached subtree");
    }

    /// Check that the `Aabb`s in the `Bvh` are tight, which means, that parent `Aabb`s are not
    /// larger than they should be. This function checks, whether the children of node `node_index`
    /// lie inside `outer_aabb`.
    pub fn assert_tight_subtree(&self, node_index: usize, outer_aabb: &Aabb<T, D>)
    where
        T: Float + SimdPartialOrd + ClosedSub + Signed,
    {
        if let BvhNode::Node {
            child_l_index,
            child_l_aabb,
            child_r_index,
            child_r_aabb,
            ..
        } = self.nodes[node_index]
        {
            let joint_aabb = child_l_aabb.join(&child_r_aabb);
            assert!(joint_aabb.relative_eq(outer_aabb, T::epsilon()));
            self.assert_tight_subtree(child_l_index, &child_l_aabb);
            self.assert_tight_subtree(child_r_index, &child_r_aabb);
        }
    }

    /// Check that the `Aabb`s in the `Bvh` are tight, which means, that parent `Aabb`s are not
    /// larger than they should be.
    pub fn assert_tight(&self)
    where
        T: SimdPartialOrd + Float + ClosedSub + Signed,
    {
        // When starting to check whether the `Bvh` is tight, we cannot provide a minimum
        // outer `Aabb`, therefore we compute the correct one in this instance.
        if let BvhNode::Node {
            child_l_aabb,
            child_r_aabb,
            ..
        } = self.nodes[0]
        {
            let joint_aabb = child_l_aabb.join(&child_r_aabb);
            self.assert_tight_subtree(0, &joint_aabb);
        }
    }
}

impl<T, const D: usize> BoundingHierarchy<T, D> for Bvh<T, D>
where
    T: Scalar
        + Copy
        + FromPrimitive
        + ToPrimitive
        + Float
        + ClosedSub
        + ClosedAdd
        + ClosedMul
        + SimdPartialOrd
        + std::fmt::Display,
{
    fn build<Shape: BHShape<T, D>>(shapes: &mut [Shape]) -> Bvh<T, D> {
        Bvh::build(shapes)
    }

    fn traverse<'a, Shape: Bounded<T, D>>(
        &'a self,
        ray: &Ray<T, D>,
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
    use crate::testbase::{build_some_bh, traverse_some_bh, TBvh3, TBvhNode3};

    #[test]
    /// Tests whether the building procedure succeeds in not failing.
    fn test_build_bvh() {
        build_some_bh::<TBvh3>();
    }

    #[test]
    /// Runs some primitive tests for intersections of a ray with a fixed scene given as a [`Bvh`].
    fn test_traverse_bvh() {
        traverse_some_bh::<TBvh3>();
    }

    #[test]
    /// Verify contents of the bounding hierarchy for a fixed scene structure
    fn test_bvh_shape_indices() {
        use std::collections::HashSet;

        let (all_shapes, bh) = build_some_bh::<TBvh3>();

        // It should find all shape indices.
        let expected_shapes: HashSet<_> = (0..all_shapes.len()).collect();
        let mut found_shapes = HashSet::new();

        for node in bh.nodes.iter() {
            match *node {
                TBvhNode3::Node { .. } => {
                    assert_eq!(node.shape_index(), None);
                }
                TBvhNode3::Leaf { .. } => {
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
    use crate::testbase::{
        build_1200_triangles_bh, build_120k_triangles_bh, build_12k_triangles_bh,
        intersect_1200_triangles_bh, intersect_120k_triangles_bh, intersect_12k_triangles_bh,
        intersect_bh, load_sponza_scene, TBvh3,
    };

    #[bench]
    /// Benchmark the construction of a [`Bvh`] with 1,200 triangles.
    fn bench_build_1200_triangles_bvh(b: &mut ::test::Bencher) {
        build_1200_triangles_bh::<TBvh3>(b);
    }

    #[bench]
    /// Benchmark the construction of a [`Bvh`] with 12,000 triangles.
    fn bench_build_12k_triangles_bvh(b: &mut ::test::Bencher) {
        build_12k_triangles_bh::<TBvh3>(b);
    }

    #[bench]
    /// Benchmark the construction of a [`Bvh`] with 120,000 triangles.
    fn bench_build_120k_triangles_bvh(b: &mut ::test::Bencher) {
        build_120k_triangles_bh::<TBvh3>(b);
    }

    #[bench]
    /// Benchmark the construction of a [`Bvh`] for the Sponza scene.
    fn bench_build_sponza_bvh(b: &mut ::test::Bencher) {
        let (mut triangles, _) = load_sponza_scene();
        b.iter(|| {
            TBvh3::build(&mut triangles);
        });
    }

    #[bench]
    /// Benchmark intersecting 1,200 triangles using the recursive [`Bvh`].
    fn bench_intersect_1200_triangles_bvh(b: &mut ::test::Bencher) {
        intersect_1200_triangles_bh::<TBvh3>(b);
    }

    #[bench]
    /// Benchmark intersecting 12,000 triangles using the recursive [`Bvh`].
    fn bench_intersect_12k_triangles_bvh(b: &mut ::test::Bencher) {
        intersect_12k_triangles_bh::<TBvh3>(b);
    }

    #[bench]
    /// Benchmark intersecting 120,000 triangles using the recursive [`Bvh`].
    fn bench_intersect_120k_triangles_bvh(b: &mut ::test::Bencher) {
        intersect_120k_triangles_bh::<TBvh3>(b);
    }

    #[bench]
    /// Benchmark the traversal of a [`Bvh`] with the Sponza scene.
    fn bench_intersect_sponza_bvh(b: &mut ::test::Bencher) {
        let (mut triangles, bounds) = load_sponza_scene();
        let bvh = TBvh3::build(&mut triangles);
        intersect_bh(&bvh, &triangles, &bounds, b)
    }
}
