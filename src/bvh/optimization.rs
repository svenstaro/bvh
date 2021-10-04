//! This module defines the optimization function for the [`BVH`].
//! By passing the indices of shapes that have changed, the function determines possible
//! tree rotations and optimizes the BVH using a SAH.
//! Based on http://www.cs.utah.edu/~thiago/papers/rotations.pdf
//!
//! [`BVH`]: struct.BVH.html
//!

use crate::aabb::AABB;
use crate::bounding_hierarchy::BHShape;
use crate::bvh::*;

use log::info;
use rand::{thread_rng, Rng};
use std::collections::HashSet;

// TODO Consider: Instead of getting the scene's shapes passed, let leaf nodes store an AABB
// that is updated from the outside, perhaps by passing not only the indices of the changed
// shapes, but also their new AABBs into optimize().
// TODO Consider: Stop updating AABBs upwards the tree once an AABB didn't get changed.

#[derive(Eq, PartialEq, Hash, Copy, Clone, Debug)]
#[allow(clippy::upper_case_acronyms)]
enum OptimizationIndex {
    Refit(usize),
    FixAABBs(usize),
}

impl OptimizationIndex {
    fn index(&self) -> usize {
        match *self {
            OptimizationIndex::Refit(index) | OptimizationIndex::FixAABBs(index) => index,
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct NodeData {
    index: usize,
    aabb: AABB,
}

impl BVHNode {
    // Get the grandchildren's NodeData.
    fn get_children_node_data(&self) -> Option<(NodeData, NodeData)> {
        match *self {
            BVHNode::Node {
                child_l_index,
                child_l_aabb,
                child_r_index,
                child_r_aabb,
                ..
            } => Some((
                NodeData {
                    index: child_l_index,
                    aabb: child_l_aabb,
                },
                NodeData {
                    index: child_r_index,
                    aabb: child_r_aabb,
                },
            )),
            BVHNode::Leaf { .. } => None,
        }
    }
}

impl BVH {
    /// Optimizes the `BVH` by batch-reorganizing updated nodes.
    /// Based on https://github.com/jeske/SimpleScene/blob/master/SimpleScene/Util/ssBVH/ssBVH.cs
    ///
    /// Needs all the scene's shapes, plus the indices of the shapes that were updated.
    ///
    pub fn optimize<Shape: BHShape>(
        &mut self,
        refit_shape_indices: &HashSet<usize>,
        shapes: &[Shape],
    ) {
        // `refit_node_indices` will contain the indices of the leaf nodes
        // that reference the given shapes, sorted by their depth
        // in increasing order.
        let mut refit_node_indices: Vec<_> = {
            let mut raw_indices = refit_shape_indices
                .iter()
                .map(|x| shapes[*x].bh_node_index())
                .collect::<Vec<_>>();

            // Sorts the Vector to have the greatest depth nodes last.
            raw_indices.sort_by(|a, b| {
                let depth_a = self.nodes[*a].depth();
                let depth_b = self.nodes[*b].depth();
                depth_a.cmp(&depth_b)
            });

            raw_indices
                .iter()
                .map(|x| OptimizationIndex::Refit(*x))
                .collect()
        };

        // As long as we have refit nodes left, take the list of refit nodes
        // with the greatest depth (sweep nodes) and try to rotate them all.
        while !refit_node_indices.is_empty() {
            let mut sweep_node_indices = Vec::new();
            let max_depth = {
                let last_node_index = refit_node_indices.last().unwrap();
                self.nodes[last_node_index.index()].depth()
            };
            while !refit_node_indices.is_empty() {
                let last_node_depth = {
                    let last_node_index = refit_node_indices.last().unwrap();
                    self.nodes[last_node_index.index()].depth()
                };
                if last_node_depth == max_depth {
                    sweep_node_indices.push(refit_node_indices.pop().unwrap());
                } else {
                    break;
                }
            }

            info!(
                "{} sweep nodes, depth {}.",
                sweep_node_indices.len(),
                max_depth
            );

            // Try to find a useful tree rotation with all previously found nodes.
            for sweep_node_index in sweep_node_indices {
                // TODO There might be multithreading potential here
                // In order to have threads working seperately without having to write on
                // the nodes vector (which would lock other threads),
                // write the results of a thread into a small data structure.
                // Apply the changes to the nodes vector represented by the data structure
                // in a quick, sequential loop after all threads finished their work.
                let new_refit_node_index = match sweep_node_index {
                    OptimizationIndex::Refit(index) => self.update(index, shapes),
                    OptimizationIndex::FixAABBs(index) => self.fix_aabbs(index, shapes),
                };

                // Instead of finding a useful tree rotation, we found another node
                // that we should check, so we add its index to the refit_node_indices.
                if let Some(index) = new_refit_node_index {
                    assert!({
                        let new_node_depth = self.nodes[index.index()].depth();
                        new_node_depth == max_depth - 1
                    });
                    refit_node_indices.push(index);
                }
            }
        }
    }

    /// This method is called for each node which has been modified and needs to be updated.
    /// If the specified node is a grandparent, then try to optimize the `BVH` by rotating its
    /// children.
    fn update<Shape: BHShape>(
        &mut self,
        node_index: usize,
        shapes: &[Shape],
    ) -> Option<OptimizationIndex> {
        info!("   [{}]\t", node_index);

        match self.nodes[node_index] {
            BVHNode::Leaf {
                parent_index,
                shape_index,
                ..
            } => {
                // The current node is a leaf.
                info!(
                    "Leaf node. Queueing parent ({}). {}.",
                    parent_index,
                    shapes[shape_index].aabb()
                );
                Some(OptimizationIndex::Refit(parent_index))
            }
            BVHNode::Node {
                parent_index,
                child_l_index,
                child_r_index,
                ..
            } => {
                // The current node is a parent.
                if let (
                    &BVHNode::Leaf {
                        shape_index: shape_l_index,
                        ..
                    },
                    &BVHNode::Leaf {
                        shape_index: shape_r_index,
                        ..
                    },
                ) = (&self.nodes[child_l_index], &self.nodes[child_r_index])
                {
                    // The current node is a final parent. Update its `AABB`s, because at least
                    // one of its children was updated and queue its parent for refitting.
                    if let BVHNode::Node {
                        ref mut child_l_aabb,
                        ref mut child_r_aabb,
                        ..
                    } = self.nodes[node_index]
                    {
                        *child_l_aabb = shapes[shape_l_index].aabb();
                        *child_r_aabb = shapes[shape_r_index].aabb();
                        info!("Setting {} from {}", child_l_aabb, child_l_index);
                        info!("\tand {} from {}.", child_r_aabb, child_r_index);
                        return Some(OptimizationIndex::Refit(parent_index));
                    }
                    unreachable!();
                }

                // The current node is a grandparent and can be optimized by rotating.
                self.try_rotate(node_index, shapes)
            }
        }
    }

    fn find_better_rotation(
        &self,
        child_l_index: usize,
        child_l_aabb: &AABB,
        child_r_index: usize,
        child_r_aabb: &AABB,
    ) -> Option<(usize, usize)> {
        // Get indices and `AABB`s of all grandchildren.
        let left_children_nodes = self.nodes[child_l_index].get_children_node_data();
        let right_children_nodes = self.nodes[child_r_index].get_children_node_data();

        // Contains the surface area that would result from applying the currently favored rotation.
        // The rotation with the smallest surface area will be applied in the end.
        // The value is calculated by `child_l_aabb.surface_area() + child_r_aabb.surface_area()`.
        let mut best_surface_area = child_l_aabb.surface_area() + child_r_aabb.surface_area();

        // Stores the rotation that would result in the surface area `best_surface_area`,
        // thus being the favored rotation that will be executed after considering all rotations.
        let mut best_rotation: Option<(usize, usize)> = None;
        {
            let mut consider_rotation = |new_rotation: (usize, usize), surface_area: f32| {
                if surface_area < best_surface_area {
                    best_surface_area = surface_area;
                    best_rotation = Some(new_rotation);
                }
            };

            // Child to grandchild rotations
            if let Some((child_rl, child_rr)) = right_children_nodes {
                let surface_area_l_rl =
                    child_rl.aabb.surface_area() + child_l_aabb.join(&child_rr.aabb).surface_area();
                consider_rotation((child_l_index, child_rl.index), surface_area_l_rl);
                let surface_area_l_rr =
                    child_rr.aabb.surface_area() + child_l_aabb.join(&child_rl.aabb).surface_area();
                consider_rotation((child_l_index, child_rr.index), surface_area_l_rr);
            }
            if let Some((child_ll, child_lr)) = left_children_nodes {
                let surface_area_r_ll =
                    child_ll.aabb.surface_area() + child_r_aabb.join(&child_lr.aabb).surface_area();
                consider_rotation((child_r_index, child_ll.index), surface_area_r_ll);
                let surface_area_r_lr =
                    child_lr.aabb.surface_area() + child_r_aabb.join(&child_ll.aabb).surface_area();
                consider_rotation((child_r_index, child_lr.index), surface_area_r_lr);

                // Grandchild to grandchild rotations
                if let Some((child_rl, child_rr)) = right_children_nodes {
                    let surface_area_ll_rl = child_rl.aabb.join(&child_lr.aabb).surface_area()
                        + child_ll.aabb.join(&child_rr.aabb).surface_area();
                    consider_rotation((child_ll.index, child_rl.index), surface_area_ll_rl);
                    let surface_area_ll_rr = child_ll.aabb.join(&child_rl.aabb).surface_area()
                        + child_lr.aabb.join(&child_rr.aabb).surface_area();
                    consider_rotation((child_ll.index, child_rr.index), surface_area_ll_rr);
                }
            }
        }
        best_rotation
    }

    /// Checks if there is a way to rotate a child and a grandchild (or two grandchildren) of
    /// the given node (specified by `node_index`) that would improve the `BVH`.
    /// If there is, the best rotation found is performed.
    ///
    /// # Preconditions
    ///
    /// This function requires that the subtree at `node_index` has correct `AABB`s.
    ///
    /// # Returns
    ///
    /// `Some(index_of_node)` if a new node was found that should be used for optimization.
    ///
    fn try_rotate<Shape: BHShape>(
        &mut self,
        node_index: usize,
        shapes: &[Shape],
    ) -> Option<OptimizationIndex> {
        let (parent_index, child_l_index, child_r_index) = if let BVHNode::Node {
            parent_index,
            child_l_index,
            child_r_index,
            ..
        } = self.nodes[node_index]
        {
            (parent_index, child_l_index, child_r_index)
        } else {
            unreachable!()
        };

        // Recalculate `AABB`s for the children since at least one of them changed.  Don't update
        // the `AABB`s in the node yet because they're still subject to change during potential
        // upcoming rotations.
        let child_l_aabb = self.nodes[child_l_index].get_node_aabb(shapes);
        let child_r_aabb = self.nodes[child_r_index].get_node_aabb(shapes);

        let best_rotation =
            self.find_better_rotation(child_l_index, &child_l_aabb, child_r_index, &child_r_aabb);

        if let Some((rotation_node_a, rotation_node_b)) = best_rotation {
            self.rotate(rotation_node_a, rotation_node_b, shapes);

            // Update the children's children `AABB`s and the children `AABB`s of node.
            self.fix_children_and_own_aabbs(node_index, shapes);

            // Return parent node's index for upcoming refitting,
            // since this node just changed its `AABB`.
            if node_index != 0 {
                Some(OptimizationIndex::Refit(parent_index))
            } else {
                None
            }
        } else {
            info!("    No useful rotation.");
            // Set the correct children `AABB`s, which have been computed earlier.
            *self.nodes[node_index].child_l_aabb_mut() = child_l_aabb;
            *self.nodes[node_index].child_r_aabb_mut() = child_r_aabb;

            // Only execute the following block, if `node_index` does not reference the root node.
            if node_index != 0 {
                // Even with no rotation being useful for this node, a parent node's rotation
                // could be beneficial, so queue the parent *sometimes*. For reference see:
                // https://github.com/jeske/SimpleScene/blob/master/SimpleScene/Util/ssBVH/ssBVH_Node.cs#L307
                // TODO Evaluate whether this is a smart thing to do.
                let mut rng = thread_rng();
                if rng.gen_bool(0.01) {
                    Some(OptimizationIndex::Refit(parent_index))
                } else {
                    // Otherwise, we still have to fix the parent's AABBs
                    Some(OptimizationIndex::FixAABBs(parent_index))
                }
            } else {
                None
            }
        }
    }

    /// Sets child_l_aabb and child_r_aabb of a BVHNode::Node to match its children,
    /// right after updating the children themselves. Not recursive.
    fn fix_children_and_own_aabbs<Shape: BHShape>(&mut self, node_index: usize, shapes: &[Shape]) {
        let (child_l_index, child_r_index) = if let BVHNode::Node {
            child_l_index,
            child_r_index,
            ..
        } = self.nodes[node_index]
        {
            (child_l_index, child_r_index)
        } else {
            unreachable!()
        };

        self.fix_aabbs(child_l_index, shapes);
        self.fix_aabbs(child_r_index, shapes);

        *self.nodes[node_index].child_l_aabb_mut() =
            self.nodes[child_l_index].get_node_aabb(shapes);
        *self.nodes[node_index].child_r_aabb_mut() =
            self.nodes[child_r_index].get_node_aabb(shapes);
    }

    /// Updates `child_l_aabb` and `child_r_aabb` of the `BVHNode::Node`
    /// with the index `node_index` from its children.
    fn fix_aabbs<Shape: BHShape>(
        &mut self,
        node_index: usize,
        shapes: &[Shape],
    ) -> Option<OptimizationIndex> {
        match self.nodes[node_index] {
            BVHNode::Node {
                parent_index,
                child_l_index,
                child_r_index,
                ..
            } => {
                *self.nodes[node_index].child_l_aabb_mut() =
                    self.nodes[child_l_index].get_node_aabb(shapes);
                *self.nodes[node_index].child_r_aabb_mut() =
                    self.nodes[child_r_index].get_node_aabb(shapes);

                if node_index > 0 {
                    Some(OptimizationIndex::FixAABBs(parent_index))
                } else {
                    None
                }
            }
            // Don't do anything if the node is a leaf.
            _ => None,
        }
    }

    /// Switch two nodes by rewiring the involved indices (not by moving them in the nodes slice).
    /// Also updates the AABBs of the parents.
    fn rotate<Shape: BHShape>(
        &mut self,
        node_a_index: usize,
        node_b_index: usize,
        shapes: &[Shape],
    ) {
        info!("    ROTATING {} and {}", node_a_index, node_b_index);

        // Get parent indices
        let node_a_parent_index = self.nodes[node_a_index].parent();
        let node_b_parent_index = self.nodes[node_b_index].parent();

        // Get info about the nodes being a left or right child
        let node_a_is_left_child = self.node_is_left_child(node_a_index);
        let node_b_is_left_child = self.node_is_left_child(node_b_index);

        // Perform the switch
        self.connect_nodes(
            node_a_index,
            node_b_parent_index,
            node_b_is_left_child,
            shapes,
        );
        self.connect_nodes(
            node_b_index,
            node_a_parent_index,
            node_a_is_left_child,
            shapes,
        );
    }

    /// Updates the depth of a node, and sets the depth of its descendants accordingly.
    fn update_depth_recursively(&mut self, node_index: usize, new_depth: u32) {
        let children = {
            let node = &mut self.nodes[node_index];
            match *node {
                BVHNode::Node {
                    ref mut depth,
                    child_l_index,
                    child_r_index,
                    ..
                } => {
                    *depth = new_depth;
                    Some((child_l_index, child_r_index))
                }
                BVHNode::Leaf { ref mut depth, .. } => {
                    *depth = new_depth;
                    None
                }
            }
        };
        if let Some((child_l_index, child_r_index)) = children {
            self.update_depth_recursively(child_l_index, new_depth + 1);
            self.update_depth_recursively(child_r_index, new_depth + 1);
        }
    }

    fn node_is_left_child(&self, node_index: usize) -> bool {
        // Get the index of the parent.
        let node_parent_index = self.nodes[node_index].parent();
        // Get the index of te left child of the parent.
        let child_l_index = self.nodes[node_parent_index].child_l();
        child_l_index == node_index
    }

    fn connect_nodes<Shape: BHShape>(
        &mut self,
        child_index: usize,
        parent_index: usize,
        left_child: bool,
        shapes: &[Shape],
    ) {
        let child_aabb = self.nodes[child_index].get_node_aabb(shapes);
        info!("\tConnecting: {} < {}.", child_index, parent_index);
        // Set parent's child and child_aabb; and get its depth.
        let parent_depth = {
            match self.nodes[parent_index] {
                BVHNode::Node {
                    ref mut child_l_index,
                    ref mut child_r_index,
                    ref mut child_l_aabb,
                    ref mut child_r_aabb,
                    depth,
                    ..
                } => {
                    if left_child {
                        *child_l_index = child_index;
                        *child_l_aabb = child_aabb;
                    } else {
                        *child_r_index = child_index;
                        *child_r_aabb = child_aabb;
                    }
                    info!("\t  {}'s new {}", parent_index, child_aabb);
                    depth
                }
                // Assuming that our BVH is correct, the parent cannot be a leaf.
                _ => unreachable!(),
            }
        };

        // Set child's parent.
        *self.nodes[child_index].parent_mut() = parent_index;

        // Update the node's and the node's descendants' depth values.
        self.update_depth_recursively(child_index, parent_depth + 1);
    }
}

#[cfg(test)]
mod tests {
    use crate::aabb::Bounded;
    use crate::bounding_hierarchy::BHShape;
    use crate::bvh::{BVHNode, BVH};
    use crate::testbase::{
        build_some_bh, create_n_cubes, default_bounds, randomly_transform_scene, UnitBox,
    };
    use crate::Point3;
    use crate::EPSILON;
    use std::collections::HashSet;

    #[test]
    /// Tests if `optimize` does not modify a fresh `BVH`.
    fn test_optimizing_new_bvh() {
        let (shapes, mut bvh) = build_some_bh::<BVH>();
        let original_nodes = bvh.nodes.clone();

        // Query an update for all nodes.
        let refit_shape_indices: HashSet<usize> = (0..shapes.len()).collect();
        bvh.optimize(&refit_shape_indices, &shapes);

        // Assert that all nodes are the same as before the update.
        for (optimized, original) in bvh.nodes.iter().zip(original_nodes.iter()) {
            assert_eq!(optimized, original);
        }
    }

    #[test]
    /// Tests whether a BVH is still consistent after a few optimization calls.
    fn test_consistent_after_optimize() {
        let (mut shapes, mut bvh) = build_some_bh::<BVH>();
        shapes[0].pos = Point3::new(10.0, 1.0, 2.0);
        shapes[1].pos = Point3::new(-10.0, -10.0, 10.0);
        shapes[2].pos = Point3::new(-10.0, 10.0, 10.0);
        shapes[3].pos = Point3::new(-10.0, 10.0, -10.0);
        shapes[4].pos = Point3::new(11.0, 1.0, 2.0);
        shapes[5].pos = Point3::new(11.0, 2.0, 2.0);

        let refit_shape_indices = (0..6).collect();
        bvh.optimize(&refit_shape_indices, &shapes);
        bvh.assert_consistent(&shapes);
    }

    #[test]
    /// Test whether a simple update on a simple BVH yields the expected optimization result.
    fn test_optimize_simple_update() {
        let mut shapes = vec![
            UnitBox::new(0, Point3::new(-50.0, 0.0, 0.0)),
            UnitBox::new(1, Point3::new(-40.0, 0.0, 0.0)),
            UnitBox::new(2, Point3::new(50.0, 0.0, 0.0)),
        ];

        let mut bvh = BVH::build(&mut shapes);
        bvh.pretty_print();

        // Assert that SAH joined shapes #0 and #1.
        {
            let left = &shapes[0];
            let moving = &shapes[1];

            match (
                &bvh.nodes[left.bh_node_index()],
                &bvh.nodes[moving.bh_node_index()],
            ) {
                (
                    &BVHNode::Leaf {
                        parent_index: left_parent_index,
                        ..
                    },
                    &BVHNode::Leaf {
                        parent_index: moving_parent_index,
                        ..
                    },
                ) => {
                    assert_eq!(moving_parent_index, left_parent_index);
                }
                _ => panic!(),
            }
        }

        // Move the first shape so that it is closer to shape #2.
        shapes[1].pos = Point3::new(40.0, 0.0, 0.0);
        let refit_shape_indices: HashSet<usize> = (1..2).collect();
        bvh.optimize(&refit_shape_indices, &shapes);
        bvh.pretty_print();
        bvh.assert_consistent(&shapes);

        // Assert that now SAH joined shapes #1 and #2.
        {
            let moving = &shapes[1];
            let right = &shapes[2];

            match (
                &bvh.nodes[right.bh_node_index()],
                &bvh.nodes[moving.bh_node_index()],
            ) {
                (
                    &BVHNode::Leaf {
                        parent_index: right_parent_index,
                        ..
                    },
                    &BVHNode::Leaf {
                        parent_index: moving_parent_index,
                        ..
                    },
                ) => {
                    assert_eq!(moving_parent_index, right_parent_index);
                }
                _ => panic!(),
            }
        }
    }

    /// Creates a small `BVH` with 4 shapes and 7 nodes.
    fn create_predictable_bvh() -> (Vec<UnitBox>, BVH) {
        let shapes = vec![
            UnitBox::new(0, Point3::new(0.0, 0.0, 0.0)),
            UnitBox::new(1, Point3::new(2.0, 0.0, 0.0)),
            UnitBox::new(2, Point3::new(4.0, 0.0, 0.0)),
            UnitBox::new(3, Point3::new(6.0, 0.0, 0.0)),
        ];

        let nodes = vec![
            // Root node.
            BVHNode::Node {
                parent_index: 0,
                depth: 0,
                child_l_aabb: shapes[0].aabb().join(&shapes[1].aabb()),
                child_l_index: 1,
                child_r_aabb: shapes[2].aabb().join(&shapes[3].aabb()),
                child_r_index: 2,
            },
            // Depth 1 nodes.
            BVHNode::Node {
                parent_index: 0,
                depth: 1,
                child_l_aabb: shapes[0].aabb(),
                child_l_index: 3,
                child_r_aabb: shapes[1].aabb(),
                child_r_index: 4,
            },
            BVHNode::Node {
                parent_index: 0,
                depth: 1,
                child_l_aabb: shapes[2].aabb(),
                child_l_index: 5,
                child_r_aabb: shapes[3].aabb(),
                child_r_index: 6,
            },
            // Depth 2 nodes (leaves).
            BVHNode::Leaf {
                parent_index: 1,
                depth: 2,
                shape_index: 0,
            },
            BVHNode::Leaf {
                parent_index: 1,
                depth: 2,
                shape_index: 1,
            },
            BVHNode::Leaf {
                parent_index: 2,
                depth: 2,
                shape_index: 2,
            },
            BVHNode::Leaf {
                parent_index: 2,
                depth: 2,
                shape_index: 3,
            },
        ];

        (shapes, BVH { nodes })
    }

    #[test]
    fn test_connect_grandchildren() {
        let (shapes, mut bvh) = create_predictable_bvh();

        // Switch two nodes
        bvh.connect_nodes(3, 2, true, &shapes);
        bvh.connect_nodes(5, 1, true, &shapes);

        // Check if the resulting tree is as expected.
        let BVH { nodes } = bvh;

        assert_eq!(nodes[0].parent(), 0);
        assert_eq!(nodes[0].child_l(), 1);
        assert_eq!(nodes[0].child_r(), 2);

        assert_eq!(nodes[1].parent(), 0);
        assert_eq!(nodes[1].child_l(), 5);
        assert_eq!(nodes[1].child_r(), 4);

        assert_eq!(nodes[2].parent(), 0);
        assert_eq!(nodes[2].child_l(), 3);
        assert_eq!(nodes[2].child_r(), 6);

        assert_eq!(nodes[3].parent(), 2);
        assert_eq!(nodes[4].parent(), 1);
        assert_eq!(nodes[5].parent(), 1);
        assert_eq!(nodes[6].parent(), 2);

        assert!(nodes[1]
            .child_l_aabb()
            .relative_eq(&shapes[2].aabb(), EPSILON));
        assert!(nodes[1]
            .child_r_aabb()
            .relative_eq(&shapes[1].aabb(), EPSILON));
        assert!(nodes[2]
            .child_l_aabb()
            .relative_eq(&shapes[0].aabb(), EPSILON));
        assert!(nodes[2]
            .child_r_aabb()
            .relative_eq(&shapes[3].aabb(), EPSILON));
    }

    #[test]
    fn test_connect_child_grandchild() {
        let (shapes, mut bvh) = create_predictable_bvh();

        // Switch two nodes
        bvh.connect_nodes(1, 2, true, &shapes);
        bvh.connect_nodes(5, 0, true, &shapes);

        // Check if the resulting tree is as expected.
        let BVH { nodes } = bvh;

        assert_eq!(nodes[0].parent(), 0);
        assert_eq!(nodes[0].child_l(), 5);
        assert_eq!(nodes[0].child_r(), 2);

        assert_eq!(nodes[1].parent(), 2);
        assert_eq!(nodes[1].child_l(), 3);
        assert_eq!(nodes[1].child_r(), 4);

        assert_eq!(nodes[2].parent(), 0);
        assert_eq!(nodes[2].child_l(), 1);
        assert_eq!(nodes[2].child_r(), 6);

        assert_eq!(nodes[3].parent(), 1);
        assert_eq!(nodes[4].parent(), 1);
        assert_eq!(nodes[5].parent(), 0);
        assert_eq!(nodes[6].parent(), 2);

        assert!(nodes[0]
            .child_l_aabb()
            .relative_eq(&shapes[2].aabb(), EPSILON));
        assert!(nodes[2]
            .child_r_aabb()
            .relative_eq(&shapes[3].aabb(), EPSILON));
        assert!(nodes[1]
            .child_l_aabb()
            .relative_eq(&shapes[0].aabb(), EPSILON));
        assert!(nodes[1]
            .child_r_aabb()
            .relative_eq(&shapes[1].aabb(), EPSILON));
    }

    #[test]
    fn test_rotate_grandchildren() {
        let (shapes, mut bvh) = create_predictable_bvh();

        // Switch two nodes.
        bvh.rotate(3, 5, &shapes);

        // Check if the resulting tree is as expected.
        let BVH { nodes } = bvh;

        assert_eq!(nodes[0].parent(), 0);
        assert_eq!(nodes[0].child_l(), 1);
        assert_eq!(nodes[0].child_r(), 2);

        assert_eq!(nodes[1].parent(), 0);
        assert_eq!(nodes[1].child_l(), 5);
        assert_eq!(nodes[1].child_r(), 4);

        assert_eq!(nodes[2].parent(), 0);
        assert_eq!(nodes[2].child_l(), 3);
        assert_eq!(nodes[2].child_r(), 6);

        assert_eq!(nodes[3].parent(), 2);
        assert_eq!(nodes[4].parent(), 1);
        assert_eq!(nodes[5].parent(), 1);
        assert_eq!(nodes[6].parent(), 2);

        assert!(nodes[1]
            .child_l_aabb()
            .relative_eq(&shapes[2].aabb(), EPSILON));
        assert!(nodes[1]
            .child_r_aabb()
            .relative_eq(&shapes[1].aabb(), EPSILON));
        assert!(nodes[2]
            .child_l_aabb()
            .relative_eq(&shapes[0].aabb(), EPSILON));
        assert!(nodes[2]
            .child_r_aabb()
            .relative_eq(&shapes[3].aabb(), EPSILON));
    }

    #[test]
    fn test_rotate_child_grandchild() {
        let (shapes, mut bvh) = create_predictable_bvh();

        // Switch two nodes.
        bvh.rotate(1, 5, &shapes);

        // Check if the resulting tree is as expected.
        let BVH { nodes } = bvh;

        assert_eq!(nodes[0].parent(), 0);
        assert_eq!(nodes[0].child_l(), 5);
        assert_eq!(nodes[0].child_r(), 2);

        assert_eq!(nodes[1].parent(), 2);
        assert_eq!(nodes[1].child_l(), 3);
        assert_eq!(nodes[1].child_r(), 4);

        assert_eq!(nodes[2].parent(), 0);
        assert_eq!(nodes[2].child_l(), 1);
        assert_eq!(nodes[2].child_r(), 6);

        assert_eq!(nodes[3].parent(), 1);
        assert_eq!(nodes[4].parent(), 1);
        assert_eq!(nodes[5].parent(), 0);
        assert_eq!(nodes[6].parent(), 2);

        assert!(nodes[0]
            .child_l_aabb()
            .relative_eq(&shapes[2].aabb(), EPSILON));
        assert!(nodes[2]
            .child_r_aabb()
            .relative_eq(&shapes[3].aabb(), EPSILON));
        assert!(nodes[1]
            .child_l_aabb()
            .relative_eq(&shapes[0].aabb(), EPSILON));
        assert!(nodes[1]
            .child_r_aabb()
            .relative_eq(&shapes[1].aabb(), EPSILON));
    }

    #[test]
    fn test_try_rotate_child_grandchild() {
        let (mut shapes, mut bvh) = create_predictable_bvh();

        // Move the second shape.
        shapes[2].pos = Point3::new(-40.0, 0.0, 0.0);

        // Try to rotate node 2 because node 5 changed.
        bvh.try_rotate(2, &shapes);

        // Try to rotate node 0 because rotating node 2 should not have yielded a result.
        bvh.try_rotate(0, &shapes);

        // Check if the resulting tree is as expected.
        let BVH { nodes } = bvh;

        assert_eq!(nodes[0].parent(), 0);
        assert_eq!(nodes[0].child_l(), 5);
        assert_eq!(nodes[0].child_r(), 2);

        assert_eq!(nodes[1].parent(), 2);
        assert_eq!(nodes[1].child_l(), 3);
        assert_eq!(nodes[1].child_r(), 4);

        assert_eq!(nodes[2].parent(), 0);
        assert_eq!(nodes[2].child_l(), 1);
        assert_eq!(nodes[2].child_r(), 6);

        assert_eq!(nodes[3].parent(), 1);
        assert_eq!(nodes[4].parent(), 1);
        assert_eq!(nodes[5].parent(), 0);
        assert_eq!(nodes[6].parent(), 2);

        assert!(nodes[0]
            .child_l_aabb()
            .relative_eq(&shapes[2].aabb(), EPSILON));
        let right_subtree_aabb = &shapes[0]
            .aabb()
            .join(&shapes[1].aabb())
            .join(&shapes[3].aabb());
        assert!(nodes[0]
            .child_r_aabb()
            .relative_eq(right_subtree_aabb, EPSILON));

        assert!(nodes[2]
            .child_r_aabb()
            .relative_eq(&shapes[3].aabb(), EPSILON));
        assert!(nodes[1]
            .child_l_aabb()
            .relative_eq(&shapes[0].aabb(), EPSILON));
        assert!(nodes[1]
            .child_r_aabb()
            .relative_eq(&shapes[1].aabb(), EPSILON));
    }

    #[test]
    /// Test optimizing `BVH` after randomizing 50% of the shapes.
    fn test_optimize_bvh_12k_75p() {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(1_000, &bounds);

        let mut bvh = BVH::build(&mut triangles);

        // The initial BVH should be consistent.
        bvh.assert_consistent(&triangles);
        bvh.assert_tight(&triangles);

        // After moving triangles, the BVH should be inconsistent, because the shape `AABB`s do not
        // match the tree entries.
        let mut seed = 0;

        let updated = randomly_transform_scene(&mut triangles, 9_000, &bounds, None, &mut seed);
        assert!(!bvh.is_consistent(&triangles), "BVH is consistent.");

        // After fixing the `AABB` consistency should be restored.
        bvh.optimize(&updated, &triangles);
        bvh.assert_consistent(&triangles);
        bvh.assert_tight(&triangles);
    }
}

#[cfg(all(feature = "bench", test))]
mod bench {
    use crate::aabb::AABB;
    use crate::bvh::BVH;
    use crate::testbase::{
        create_n_cubes, default_bounds, intersect_bh, load_sponza_scene, randomly_transform_scene,
        Triangle,
    };

    #[bench]
    /// Benchmark randomizing 50% of the shapes in a `BVH`.
    fn bench_randomize_120k_50p(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        let mut seed = 0;

        b.iter(|| {
            randomly_transform_scene(&mut triangles, 60_000, &bounds, None, &mut seed);
        });
    }

    /// Benchmark optimizing a `BVH` with 120,000 `Triangle`s, where `percent`
    /// `Triangles` have been randomly moved.
    fn optimize_bvh_120k(percent: f32, b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        let mut bvh = BVH::build(&mut triangles);
        let num_move = (triangles.len() as f32 * percent) as usize;
        let mut seed = 0;

        b.iter(|| {
            let updated =
                randomly_transform_scene(&mut triangles, num_move, &bounds, Some(10.0), &mut seed);
            bvh.optimize(&updated, &triangles);
        });
    }

    #[bench]
    fn bench_optimize_bvh_120k_00p(b: &mut ::test::Bencher) {
        optimize_bvh_120k(0.0, b);
    }

    #[bench]
    fn bench_optimize_bvh_120k_01p(b: &mut ::test::Bencher) {
        optimize_bvh_120k(0.01, b);
    }

    #[bench]
    fn bench_optimize_bvh_120k_10p(b: &mut ::test::Bencher) {
        optimize_bvh_120k(0.1, b);
    }

    #[bench]
    fn bench_optimize_bvh_120k_50p(b: &mut ::test::Bencher) {
        optimize_bvh_120k(0.5, b);
    }

    /// Move `percent` `Triangle`s in the scene given by `triangles` and optimize the
    /// `BVH`. Iterate this procedure `iterations` times. Afterwards benchmark the performance
    /// of intersecting this scene/`BVH`.
    fn intersect_scene_after_optimize(
        triangles: &mut Vec<Triangle>,
        bounds: &AABB,
        percent: f32,
        max_offset: Option<f32>,
        iterations: usize,
        b: &mut ::test::Bencher,
    ) {
        let mut bvh = BVH::build(triangles);
        let num_move = (triangles.len() as f32 * percent) as usize;
        let mut seed = 0;

        for _ in 0..iterations {
            let updated =
                randomly_transform_scene(triangles, num_move, bounds, max_offset, &mut seed);
            bvh.optimize(&updated, triangles);
        }

        intersect_bh(&bvh, triangles, bounds, b);
    }

    #[bench]
    fn bench_intersect_120k_after_optimize_00p(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        intersect_scene_after_optimize(&mut triangles, &bounds, 0.0, None, 10, b);
    }

    #[bench]
    fn bench_intersect_120k_after_optimize_01p(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        intersect_scene_after_optimize(&mut triangles, &bounds, 0.01, None, 10, b);
    }

    #[bench]
    fn bench_intersect_120k_after_optimize_10p(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        intersect_scene_after_optimize(&mut triangles, &bounds, 0.1, None, 10, b);
    }

    #[bench]
    fn bench_intersect_120k_after_optimize_50p(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        intersect_scene_after_optimize(&mut triangles, &bounds, 0.5, None, 10, b);
    }

    /// Move `percent` `Triangle`s in the scene given by `triangles` `iterations` times.
    /// Afterwards optimize the `BVH` and benchmark the performance of intersecting this
    /// scene/`BVH`. Used to compare optimizing with rebuilding. For reference see
    /// `intersect_scene_after_optimize`.
    fn intersect_scene_with_rebuild(
        triangles: &mut Vec<Triangle>,
        bounds: &AABB,
        percent: f32,
        max_offset: Option<f32>,
        iterations: usize,
        b: &mut ::test::Bencher,
    ) {
        let num_move = (triangles.len() as f32 * percent) as usize;
        let mut seed = 0;
        for _ in 0..iterations {
            randomly_transform_scene(triangles, num_move, bounds, max_offset, &mut seed);
        }

        let bvh = BVH::build(triangles);
        intersect_bh(&bvh, triangles, bounds, b);
    }

    #[bench]
    fn bench_intersect_120k_with_rebuild_00p(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        intersect_scene_with_rebuild(&mut triangles, &bounds, 0.0, None, 10, b);
    }

    #[bench]
    fn bench_intersect_120k_with_rebuild_01p(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        intersect_scene_with_rebuild(&mut triangles, &bounds, 0.01, None, 10, b);
    }

    #[bench]
    fn bench_intersect_120k_with_rebuild_10p(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        intersect_scene_with_rebuild(&mut triangles, &bounds, 0.1, None, 10, b);
    }

    #[bench]
    fn bench_intersect_120k_with_rebuild_50p(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        intersect_scene_with_rebuild(&mut triangles, &bounds, 0.5, None, 10, b);
    }

    /// Benchmark intersecting a `BVH` for Sponza after randomly moving one `Triangle` and
    /// optimizing.
    fn intersect_sponza_after_optimize(percent: f32, b: &mut ::test::Bencher) {
        let (mut triangles, bounds) = load_sponza_scene();
        intersect_scene_after_optimize(&mut triangles, &bounds, percent, Some(0.1), 10, b);
    }

    #[bench]
    fn bench_intersect_sponza_after_optimize_00p(b: &mut ::test::Bencher) {
        intersect_sponza_after_optimize(0.0, b);
    }

    #[bench]
    fn bench_intersect_sponza_after_optimize_01p(b: &mut ::test::Bencher) {
        intersect_sponza_after_optimize(0.01, b);
    }

    #[bench]
    fn bench_intersect_sponza_after_optimize_10p(b: &mut ::test::Bencher) {
        intersect_sponza_after_optimize(0.1, b);
    }

    #[bench]
    fn bench_intersect_sponza_after_optimize_50p(b: &mut ::test::Bencher) {
        intersect_sponza_after_optimize(0.5, b);
    }

    /// Benchmark intersecting a `BVH` for Sponza after rebuilding. Used to compare optimizing
    /// with rebuilding. For reference see `intersect_sponza_after_optimize`.
    fn intersect_sponza_with_rebuild(percent: f32, b: &mut ::test::Bencher) {
        let (mut triangles, bounds) = load_sponza_scene();
        intersect_scene_with_rebuild(&mut triangles, &bounds, percent, Some(0.1), 10, b);
    }

    #[bench]
    fn bench_intersect_sponza_with_rebuild_00p(b: &mut ::test::Bencher) {
        intersect_sponza_with_rebuild(0.0, b);
    }

    #[bench]
    fn bench_intersect_sponza_with_rebuild_01p(b: &mut ::test::Bencher) {
        intersect_sponza_with_rebuild(0.01, b);
    }

    #[bench]
    fn bench_intersect_sponza_with_rebuild_10p(b: &mut ::test::Bencher) {
        intersect_sponza_with_rebuild(0.1, b);
    }

    #[bench]
    fn bench_intersect_sponza_with_rebuild_50p(b: &mut ::test::Bencher) {
        intersect_sponza_with_rebuild(0.5, b);
    }
}
