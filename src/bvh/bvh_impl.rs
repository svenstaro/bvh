//! This module defines [`Bvh`] and [`BvhNode`] and functions for building and traversing it.
//!
//! [`Bvh`]: struct.Bvh.html
//! [`BvhNode`]: struct.BvhNode.html
//!

use nalgebra::Point;

use crate::aabb::{Aabb, Bounded, IntersectsAabb};
use crate::bounding_hierarchy::{BHShape, BHValue, BoundingHierarchy};
use crate::bvh::iter::BvhTraverseIterator;
use crate::ray::Ray;
use crate::utils::joint_aabb_of_shapes;

use std::mem::MaybeUninit;

use super::{
    BvhNode, BvhNodeBuildArgs, ChildDistanceTraverseIterator, DistanceTraverseIterator, ShapeIndex,
    Shapes,
};

/// The [`Bvh`] data structure. Contains the list of [`BvhNode`]s.
///
/// [`Bvh`]: struct.Bvh.html
///
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Bvh<T: BHValue, const D: usize> {
    /// The list of nodes of the [`Bvh`].
    ///
    /// [`Bvh`]: struct.Bvh.html
    ///
    pub nodes: Vec<BvhNode<T, D>>,
}

impl<T: BHValue, const D: usize> Bvh<T, D> {
    /// Creates a new [`Bvh`] from the `shapes` slice.
    ///
    /// [`Bvh`]: struct.Bvh.html
    ///
    pub fn build<Shape: BHShape<T, D>>(shapes: &mut [Shape]) -> Bvh<T, D> {
        Self::build_with_executor(shapes, |left, right| {
            left.build();
            right.build();
        })
    }

    /// Creates a new [`Bvh`] from the `shapes` slice.
    /// The executor parameter allows you to parallelize the build of the [`Bvh`]. Using something like rayon::join.
    /// You must call either build or build_with_executor on both arguments in order to succesfully complete the build.
    ///
    /// [`Bvh`]: struct.Bvh.html
    ///
    pub fn build_with_executor<Shape: BHShape<T, D>>(
        shapes: &mut [Shape],
        executor: impl FnMut(BvhNodeBuildArgs<Shape, T, D>, BvhNodeBuildArgs<Shape, T, D>),
    ) -> Bvh<T, D> {
        if shapes.is_empty() {
            return Bvh { nodes: Vec::new() };
        }

        let mut indices = (0..shapes.len())
            .map(ShapeIndex)
            .collect::<Vec<ShapeIndex>>();
        let expected_node_count = shapes.len() * 2 - 1;
        let mut nodes = Vec::with_capacity(expected_node_count);

        let uninit_slice = unsafe {
            std::slice::from_raw_parts_mut(
                nodes.as_mut_ptr() as *mut MaybeUninit<BvhNode<T, D>>,
                expected_node_count,
            )
        };
        let shapes = Shapes::from_slice(shapes);
        let (aabb, centroid) = joint_aabb_of_shapes(&indices, &shapes);
        BvhNode::build_with_executor(
            BvhNodeBuildArgs {
                shapes: &shapes,
                indices: &mut indices,
                nodes: uninit_slice,
                parent_index: 0,
                depth: 0,
                node_index: 0,
                aabb_bounds: aabb,
                centroid_bounds: centroid,
            },
            executor,
        );

        // SAFETY
        // The vec is allocated with this capacity above and is only mutated through slice methods so
        // it is guaranteed that the allocated size has not changed.
        unsafe {
            nodes.set_len(expected_node_count);
        }
        Bvh { nodes }
    }

    /// Traverses the [`Bvh`].
    /// Returns a subset of `shapes`, in which the [`Aabb`]s of the elements were hit by [`Ray`].
    ///
    /// [`Bvh`]: struct.Bvh.html
    /// [`Aabb`]: ../aabb/struct.Aabb.html
    ///
    pub fn traverse<'a, Query: IntersectsAabb<T, D>, Shape: Bounded<T, D>>(
        &'a self,
        query: &Query,
        shapes: &'a [Shape],
    ) -> Vec<&'a Shape> {
        if self.nodes.is_empty() {
            // There won't be a 0th node_index.
            return Vec::new();
        }
        let mut indices = Vec::new();
        BvhNode::traverse_recursive(&self.nodes, 0, shapes, query, &mut indices);
        indices
            .iter()
            .map(|index| &shapes[*index])
            .collect::<Vec<_>>()
    }

    /// Creates a [`BvhTraverseIterator`] to traverse the [`Bvh`].
    /// Returns a subset of `shapes`, in which the [`Aabb`]s of the elements for which
    /// [`IntersectsAabb::intersects_aabb`] returns `true`.
    ///
    /// [`Bvh`]: struct.Bvh.html
    /// [`Aabb`]: ../aabb/struct.Aabb.html
    ///
    pub fn traverse_iterator<'bvh, 'shape, Query: IntersectsAabb<T, D>, Shape: Bounded<T, D>>(
        &'bvh self,
        query: &'bvh Query,
        shapes: &'shape [Shape],
    ) -> BvhTraverseIterator<'bvh, 'shape, T, D, Query, Shape> {
        BvhTraverseIterator::new(self, query, shapes)
    }

    /// Creates a [`DistanceTraverseIterator`] to traverse the [`Bvh`].
    /// Returns a subset of [`shape`], in which the [`Aabb`]s of the elements were hit by [`Ray`].
    /// Return in order from nearest to farthest for ray.
    ///
    /// Time complexity: for first `O(log(n))`, for all `O(n*log(n))`
    ///
    /// [`Bvh`]: struct.Bvh.html
    /// [`Aabb`]: ../aabb/struct.AABB.html
    ///
    pub fn nearest_traverse_iterator<'bvh, 'shape, Shape: Bounded<T, D>>(
        &'bvh self,
        ray: &'bvh Ray<T, D>,
        shapes: &'shape [Shape],
    ) -> DistanceTraverseIterator<'bvh, 'shape, T, D, Shape, true> {
        DistanceTraverseIterator::new(self, ray, shapes)
    }

    /// Creates a [`DistanceTraverseIterator`] to traverse the [`Bvh`].
    /// Returns a subset of [`Shape`], in which the [`Aabb`]s of the elements were hit by [`Ray`].
    /// Return in order from farthest to nearest for ray.
    ///
    /// Time complexity: for first `O(log(n))`, for all `O(n*log(n))`.
    ///
    /// [`Bvh`]: struct.Bvh.html
    /// [`Aabb`]: ../aabb/struct.AABB.html
    ///
    pub fn farthest_traverse_iterator<'bvh, 'shape, Shape: Bounded<T, D>>(
        &'bvh self,
        ray: &'bvh Ray<T, D>,
        shapes: &'shape [Shape],
    ) -> DistanceTraverseIterator<'bvh, 'shape, T, D, Shape, false> {
        DistanceTraverseIterator::new(self, ray, shapes)
    }

    /// Creates a [`ChildDistanceTraverseIterator`] to traverse the [`Bvh`].
    /// Returns a subset of [`shape`], in which the [`Aabb`]s of the elements were hit by [`Ray`].
    /// Return in order from nearest to farthest for ray.
    ///
    /// This is a best-effort function that orders interior parent nodes before ordering child
    /// nodes, so the output is not necessarily perfectly sorted.
    ///
    /// So the output is not necessarily perfectly sorted if the children of any node overlap.
    ///
    /// Time complexity: for first `O(log(n))`, for all `O(n)`.
    ///
    /// [`Bvh`]: struct.Bvh.html
    /// [`Aabb`]: ../aabb/struct.AABB.html
    ///
    pub fn nearest_child_traverse_iterator<'bvh, 'shape, Shape: Bounded<T, D>>(
        &'bvh self,
        ray: &'bvh Ray<T, D>,
        shapes: &'shape [Shape],
    ) -> ChildDistanceTraverseIterator<'bvh, 'shape, T, D, Shape, true> {
        ChildDistanceTraverseIterator::new(self, ray, shapes)
    }

    /// Creates a [`ChildDistanceTraverseIterator`] to traverse the [`Bvh`].
    /// Returns a subset of [`Shape`], in which the [`Aabb`]s of the elements were hit by [`Ray`].
    /// Return in order from farthest to nearest for ray.
    ///
    /// This is a best-effort function that orders interior parent nodes before ordering child
    /// nodes, so the output is not necessarily perfectly sorted.
    ///
    /// So the output is not necessarily perfectly sorted if the children of any node overlap.
    ///
    /// Time complexity: for first `O(log(n))`, for all `O(n)`.
    ///
    /// [`Bvh`]: struct.Bvh.html
    /// [`Aabb`]: ../aabb/struct.AABB.html
    ///
    pub fn farthest_child_traverse_iterator<'bvh, 'shape, Shape: Bounded<T, D>>(
        &'bvh self,
        ray: &'bvh Ray<T, D>,
        shapes: &'shape [Shape],
    ) -> ChildDistanceTraverseIterator<'bvh, 'shape, T, D, Shape, false> {
        ChildDistanceTraverseIterator::new(self, ray, shapes)
    }

    /// Traverses the [`Bvh`].
    /// Returns a subset of `shapes` which are candidates for being the closest to `point`.
    ///
    ///
    /// [`Bvh`]: struct.Bvh.html
    /// [`Aabb`]: ../aabb/struct.Aabb.html
    ///
    pub fn nearest_candidates<'a, Shape: Bounded<T, D>>(
        &self,
        origin: &Point<T, D>,
        shapes: &'a [Shape],
    ) -> Vec<&'a Shape>
    where
        Self: std::marker::Sized,
    {
        let mut indices = Vec::new();
        let mut best_min_distance = T::max_value();
        let mut best_max_distance = T::max_value();
        BvhNode::nearest_candidates_recursive(
            &self.nodes,
            0,
            origin,
            shapes,
            &mut indices,
            &mut best_min_distance,
            &mut best_max_distance,
        );

        indices
            .into_iter()
            // Filter out shapes that are too far but couldn't be pruned before.
            .filter(|(_, node_min)| *node_min <= best_max_distance)
            .map(|(i, _)| &shapes[i])
            .collect()
    }

    /// Prints the [`Bvh`] in a tree-like visualization.
    ///
    /// [`Bvh`]: struct.Bvh.html
    ///
    pub fn pretty_print(&self) {
        let nodes = &self.nodes;
        fn print_node<T: BHValue, const D: usize>(
            nodes: &[BvhNode<T, D>],
            node_index: usize,
            depth: usize,
        ) {
            match nodes[node_index] {
                BvhNode::Node {
                    child_l_index,
                    child_r_index,
                    child_l_aabb,
                    child_r_aabb,
                    ..
                } => {
                    let padding: String = " ".repeat(depth);
                    println!("{}child_l {}", padding, child_l_aabb);
                    print_node(nodes, child_l_index, depth + 1);
                    println!("{}child_r {}", padding, child_r_aabb);
                    print_node(nodes, child_r_index, depth + 1);
                }
                BvhNode::Leaf { shape_index, .. } => {
                    let padding: String = " ".repeat(depth);
                    println!("{}shape\t{:?}", padding, shape_index);
                }
            }
        }
        print_node(nodes, 0, 0);
    }

    /// Verifies that the node at index `node_index` lies inside `expected_outer_aabb`,
    /// its parent index is equal to `expected_parent_index`, its depth is equal to
    /// `expected_depth`. Increases `node_count` by the number of visited nodes.
    fn is_consistent_subtree<Shape: BHShape<T, D>>(
        &self,
        node_index: usize,
        expected_parent_index: usize,
        expected_outer_aabb: &Aabb<T, D>,
        node_count: &mut usize,
        shapes: &[Shape],
    ) -> bool {
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
                    node_count,
                    shapes,
                );
                let right_subtree_consistent = self.is_consistent_subtree(
                    child_r_index,
                    node_index,
                    &child_r_aabb,
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
    pub fn is_consistent<Shape: BHShape<T, D>>(&self, shapes: &[Shape]) -> bool {
        if self.nodes.is_empty() {
            // There is no node_index=0.
            return true;
        }

        // The root node of the bvh is not bounded by anything.
        let space = Aabb::infinite();

        // The counter for all nodes.
        let mut node_count = 0;
        let subtree_consistent = self.is_consistent_subtree(0, 0, &space, &mut node_count, shapes);

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
        node_count: &mut usize,
        shapes: &[Shape],
    ) where
        T: std::fmt::Display,
    {
        *node_count += 1;
        let node = &self.nodes[node_index];

        let parent = node.parent();
        assert_eq!(
            expected_parent_index, parent,
            "Wrong parent index. Expected: {expected_parent_index}; Actual: {parent}"
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
                    node_count,
                    shapes,
                );
                self.assert_consistent_subtree(
                    child_r_index,
                    node_index,
                    &child_r_aabb,
                    node_count,
                    shapes,
                );
            }
            BvhNode::Leaf { shape_index, .. } => {
                let shape_aabb = shapes[shape_index].aabb();
                assert!(
                    expected_outer_aabb.approx_contains_aabb_eps(&shape_aabb, T::epsilon()),
                    "Shape's Aabb lies outside the expected bounds.\n\tBounds: {expected_outer_aabb}\n\tShape: {shape_aabb}"
                );
            }
        }
    }

    /// Assert version of `is_consistent`.
    pub fn assert_consistent<Shape: BHShape<T, D>>(&self, shapes: &[Shape])
    where
        T: std::fmt::Display,
    {
        if self.nodes.is_empty() {
            // There is no node_index=0.
            return;
        }

        // The root node of the bvh is not bounded by anything.
        let space = Aabb::infinite();

        // The counter for all nodes.
        let mut node_count = 0;
        self.assert_consistent_subtree(0, 0, &space, &mut node_count, shapes);

        // Check if all nodes have been counted from the root node.
        // If this is false, it means we have a detached subtree.
        assert_eq!(node_count, self.nodes.len(), "Detached subtree");
    }

    /// Check that the `Aabb`s in the `Bvh` are tight, which means, that parent `Aabb`s are not
    /// larger than they should be. This function checks, whether the children of node `node_index`
    /// lie inside `outer_aabb`.
    pub fn assert_tight_subtree(&self, node_index: usize, outer_aabb: &Aabb<T, D>) {
        if let BvhNode::Node {
            child_l_index,
            child_l_aabb,
            child_r_index,
            child_r_aabb,
            ..
        } = self.nodes[node_index]
        {
            let joint_aabb = child_l_aabb.join(&child_r_aabb);
            // Aabb's are joined by selecting the min/max of coordinates on each axis,
            // which doesn't involve rounding errors. This is equally true here and
            // inside the BVH implementation. Therefore, we can assert exact equality.
            assert_eq!(&joint_aabb, outer_aabb);
            self.assert_tight_subtree(child_l_index, &child_l_aabb);
            self.assert_tight_subtree(child_r_index, &child_r_aabb);
        }
    }

    /// Check that the `Aabb`s in the `Bvh` are tight, which means, that parent `Aabb`s are not
    /// larger than they should be.
    pub fn assert_tight(&self) {
        if self.nodes.is_empty() {
            // There is no node_index=0.
            return;
        }
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

impl<T: BHValue + std::fmt::Display, const D: usize> BoundingHierarchy<T, D> for Bvh<T, D> {
    fn build<Shape: BHShape<T, D>>(shapes: &mut [Shape]) -> Bvh<T, D> {
        Bvh::build(shapes)
    }

    fn traverse<'a, Query: IntersectsAabb<T, D>, Shape: Bounded<T, D>>(
        &'a self,
        query: &Query,
        shapes: &'a [Shape],
    ) -> Vec<&'a Shape> {
        self.traverse(query, shapes)
    }

    fn nearest_candidates<'a, Shape: BHShape<T, D>>(
        &'a self,
        query: &Point<T, D>,
        shapes: &'a [Shape],
    ) -> Vec<&Shape> {
        self.nearest_candidates(query, shapes)
    }

    fn pretty_print(&self) {
        self.pretty_print();
    }

    fn build_with_executor<
        Shape: BHShape<T, D>,
        Executor: FnMut(BvhNodeBuildArgs<'_, Shape, T, D>, BvhNodeBuildArgs<'_, Shape, T, D>),
    >(
        shapes: &mut [Shape],
        executor: Executor,
    ) -> Self {
        Bvh::build_with_executor(shapes, executor)
    }
}

/// Rayon based executor
#[cfg(feature = "rayon")]
pub fn rayon_executor<S, T: Send + BHValue, const D: usize>(
    left: BvhNodeBuildArgs<S, T, D>,
    right: BvhNodeBuildArgs<S, T, D>,
) where
    S: BHShape<T, D> + Send,
{
    // 64 was found experimentally. Calling join() has overhead that makes the build slower without this.
    if left.node_count() + right.node_count() < 64 {
        left.build();
        right.build();
    } else {
        rayon::join(
            || left.build_with_executor(rayon_executor),
            || right.build_with_executor(rayon_executor),
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        bounding_hierarchy::BoundingHierarchy,
        testbase::{
            build_empty_bh, build_some_bh, nearest_candidates_some_bh, traverse_some_bh, TBvh3,
            TBvhNode3, TPoint3, TRay3, TVector3, UnitBox,
        },
    };

    #[test]
    /// Tests whether the building procedure succeeds in not failing.
    fn test_build_bvh() {
        build_some_bh::<TBvh3>();
    }

    #[test]
    fn test_empty_bvh_is_consistent() {
        let (shapes, bvh) = build_empty_bh::<TBvh3>();
        bvh.assert_consistent(&shapes);
        assert!(bvh.is_consistent(&shapes));
    }

    #[test]
    fn test_empty_bvh_is_tight() {
        let (_, bvh) = build_empty_bh::<TBvh3>();
        bvh.assert_tight();
    }

    #[test]
    /// Runs some primitive tests for intersections of a ray with a fixed scene given as a [`Bvh`].
    fn test_traverse_bvh() {
        traverse_some_bh::<TBvh3>();
    }

    #[test]
    /// Runs some primitive tests for distance query of a point with a fixed scene given as a [`Bvh`].
    fn test_nearest_candidate_bvh() {
        nearest_candidates_some_bh::<TBvh3>();
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

    #[test]
    #[cfg(feature = "rayon")]
    /// Tests whether the building procedure succeeds in not failing.
    fn test_build_bvh_rayon() {
        use crate::testbase::build_some_bh_rayon;

        build_some_bh_rayon::<TBvh3>();
    }

    #[test]
    #[cfg(feature = "rayon")]
    /// Runs some primitive tests for intersections of a ray with a fixed scene given as a [`Bvh`].
    fn test_traverse_bvh_rayon() {
        use crate::testbase::traverse_some_bh_rayon;

        traverse_some_bh_rayon::<TBvh3>();
    }

    #[test]
    #[cfg(feature = "rayon")]
    /// Verify contents of the bounding hierarchy for a fixed scene structure
    fn test_bvh_shape_indices_rayon() {
        use std::collections::HashSet;

        use crate::testbase::build_some_bh_rayon;

        let (all_shapes, bh) = build_some_bh_rayon::<TBvh3>();

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

    /// A single-node BVH is special, since the root node is a leaf node. Make sure
    /// the root node isn't unconditionally returned when it isn't intersected.
    #[test]
    fn test_traverse_one_node_bvh_no_intersection() {
        let mut boxes = vec![UnitBox::new(0, TPoint3::new(0.0, 1.0, 2.0))];
        let ray = TRay3::new(TPoint3::new(0.0, 0.0, 0.0), TVector3::new(1.0, 0.0, 0.0));
        let bvh = TBvh3::build(&mut boxes);

        assert!(bvh.traverse(&ray, &boxes).is_empty());
        assert!(bvh.traverse_iterator(&ray, &boxes).next().is_none());
        assert!(bvh.nearest_traverse_iterator(&ray, &boxes).next().is_none());
        assert!(bvh.flatten().traverse(&ray, &boxes).is_empty())
    }

    /// Make sure the root node can be returned when it is intersected.
    #[test]
    fn test_traverse_one_node_bvh_intersection() {
        let mut boxes = vec![UnitBox::new(0, TPoint3::new(10.0, 0.0, 0.0))];
        let ray = TRay3::new(TPoint3::new(0.0, 0.0, 0.0), TVector3::new(1.0, 0.0, 0.0));
        let bvh = TBvh3::build(&mut boxes);

        assert_eq!(bvh.traverse(&ray, &boxes).len(), 1);
        assert_eq!(bvh.traverse_iterator(&ray, &boxes).count(), 1);
        assert_eq!(bvh.nearest_traverse_iterator(&ray, &boxes).count(), 1);
        assert_eq!(bvh.flatten().traverse(&ray, &boxes).len(), 1)
    }
}

#[cfg(all(feature = "bench", test))]
mod bench {
    #[cfg(feature = "rayon")]
    use crate::bounding_hierarchy::BoundingHierarchy;
    use crate::testbase::{
        build_1200_triangles_bh, build_120k_triangles_bh, build_12k_triangles_bh,
        intersect_1200_triangles_bh, intersect_120k_triangles_bh, intersect_12k_triangles_bh,
        intersect_bh, load_sponza_scene, TBvh3,
    };
    #[cfg(feature = "rayon")]
    use crate::testbase::{
        build_1200_triangles_bh_rayon, build_120k_triangles_bh_rayon, build_12k_triangles_bh_rayon,
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

    #[cfg(feature = "rayon")]
    #[bench]
    /// Benchmark the construction of a `BVH` with 1,200 triangles.
    fn bench_build_1200_triangles_bvh_rayon(b: &mut ::test::Bencher) {
        build_1200_triangles_bh_rayon::<TBvh3>(b);
    }

    #[bench]
    #[cfg(feature = "rayon")]
    /// Benchmark the construction of a `BVH` with 12,000 triangles.
    fn bench_build_12k_triangles_bvh_rayon(b: &mut ::test::Bencher) {
        build_12k_triangles_bh_rayon::<TBvh3>(b);
    }

    #[bench]
    #[cfg(feature = "rayon")]
    /// Benchmark the construction of a `BVH` with 120,000 triangles.
    fn bench_build_120k_triangles_bvh_rayon(b: &mut ::test::Bencher) {
        build_120k_triangles_bh_rayon::<TBvh3>(b);
    }

    #[bench]
    #[cfg(feature = "rayon")]
    /// Benchmark the construction of a `BVH` for the Sponza scene.
    fn bench_build_sponza_bvh_rayon(b: &mut ::test::Bencher) {
        let (mut triangles, _) = load_sponza_scene();
        b.iter(|| {
            TBvh3::build_par(&mut triangles);
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
