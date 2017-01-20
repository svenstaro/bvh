//! This module defines the optimization function for the [`BVH`].
//! By passing the indices of shapes that have changed, the function determines possible
//! tree rotations and optimizes the BVH using a SAH.
//! Based on http://www.cs.utah.edu/~thiago/papers/rotations.pdf
//!
//! [`BVH`]: struct.BVH.html
//!

use bvh::*;
use bounding_hierarchy::BHShape;
use aabb::AABB;
use std::collections::HashSet;
use rand::{thread_rng, Rng};

// TODO Consider: Instead of getting the scene's shapes passed, let leaf nodes store an AABB
// that is updated from the outside, perhaps by passing not only the indices of the changed
// shapes, but also their new AABBs into optimize().

impl BVH {
    /// Optimizes the `BVH` by batch-reorganizing updated nodes.
    /// Based on https://github.com/jeske/SimpleScene/blob/master/SimpleScene/Util/ssBVH/ssBVH.cs
    ///
    /// Needs all the scene's shapes, plus the indices of the shapes that were updated.
    ///
    pub fn optimize<Shape: BHShape>(&mut self,
                                    refit_shape_indices: &HashSet<usize>,
                                    shapes: &[Shape]) {
        let mut refit_node_indices: HashSet<usize> = refit_shape_indices.iter()
            .map(|x| shapes[*x].bh_node_index())
            .collect();

        // As long as we have refit nodes left, take the list of refit nodes
        // with the highest depth (sweep nodes) and try to rotate them all
        while refit_node_indices.len() > 0 {
            let mut max_depth = 0;
            let mut sweep_node_indices = Vec::new();

            // Find max_depth and sweep_node_indices in one iteration
            for refit_node_index in refit_node_indices.iter() {
                let depth = match self.nodes[*refit_node_index] {
                    BVHNode::Node { depth, .. } => depth,
                    BVHNode::Leaf { depth, .. } => depth,
                };

                if depth > max_depth {
                    max_depth = depth;
                    sweep_node_indices.clear();
                    sweep_node_indices.push(*refit_node_index);
                } else if depth == max_depth {
                    sweep_node_indices.push(*refit_node_index);
                }
            }

            // Try to find a useful tree rotation with all nodes previously found
            for sweep_node_index in &sweep_node_indices {
                // This node does not need to be checked again
                refit_node_indices.remove(sweep_node_index);

                // TODO There might be multithreading potential here
                let new_refit_node_index = self.try_rotate(*sweep_node_index, shapes);

                // Instead of finding a useful tree rotation, we found another node
                // that we should check, so we add its index to the refit_node_indices.
                if let Some(index) = new_refit_node_index {
                    refit_node_indices.insert(index);
                }
            }
        }
    }

    /// Checks if there is a way to rotate a child and a grandchild (or two grandchildren) of
    /// the given node (specified by `node_index`) that would improve the `BVH`.
    /// If there is, the best rotation found is performed.
    /// Relies on the children and transitive children of the given node having correct AABBs.
    ///
    /// Returns Some(index_of_node) if a new node was found that should be used for optimization.
    ///
    fn try_rotate<Shape: BHShape>(&mut self, node_index: usize, shapes: &[Shape]) -> Option<usize> {
        let mut nodes = &mut self.nodes;

        let node_clone = nodes[node_index].clone();

        // Contains the surface area that would result from applying the currently favored rotation.
        // The rotation with the smallest SA will be applied in the end.
        // The value is calculated by child_l_aabb.surface_area() + child_r_aabb.surface_area()
        let mut best_surface_area = 0f32;

        // TODO Re-implement without mutability
        let mut parent_index: usize = 0;

        #[derive(Debug, Copy, Clone)]
        struct NodeData {
            index: usize,
            aabb: AABB,
        }

        // If this node is not a grandparent, update the AABB,
        // queue the parent for refitting, and bail out.
        // If it is a grandparent, calculate the current best_surface_area.
        let (child_l, child_r) = match node_clone {
            BVHNode::Node { parent, child_l, child_r, child_l_aabb, child_r_aabb, .. } => {
                if let BVHNode::Leaf { shape, .. } = nodes[child_l] {
                    let shape_l_index = shape;
                    if let BVHNode::Leaf { shape, .. } = nodes[child_r] {
                        let shape_r_index = shape;

                        // Update the AABBs saved for the children
                        // since at least one of them changed.
                        let mut node = &mut nodes[node_index];
                        match node {
                            &mut BVHNode::Node { ref mut child_l_aabb,
                                                 ref mut child_r_aabb,
                                                 .. } => {
                                *child_l_aabb = shapes[shape_l_index].aabb();
                                *child_r_aabb = shapes[shape_r_index].aabb();
                            }
                            // We know that node must be a BVHNode::Node at this point
                            _ => unreachable!(),
                        }

                        return Some(parent);
                    }
                }

                // Recalculate AABBs for the children since at least one of them changed.
                // Don't update the AABBs yet because they're still subject to change
                // during potential upcoming rotations.
                let aabb_l = BVH::get_node_aabb(&nodes[child_l].clone(), shapes);
                let aabb_r = BVH::get_node_aabb(&nodes[child_r].clone(), shapes);

                parent_index = parent;
                best_surface_area = child_l_aabb.surface_area() + child_r_aabb.surface_area();
                (NodeData {
                     index: child_l,
                     aabb: aabb_l,
                 },
                 NodeData {
                     index: child_r,
                     aabb: aabb_r,
                 })
            }
            BVHNode::Leaf { parent, .. } => {
                return Some(parent);
            }
        };

        // Stores the Rotation that would result in the surface area best_surface_area,
        // thus being the favored rotation that will be executed after considering all rotations.
        let mut best_rotation: Option<(usize, usize)> = None;
        let mut perform_grandchild_rotation = false;

        // Get the grandchildren's NodeData
        let (left_children_nodes, right_children_nodes) = {
            let get_children_node_data = |node_index: usize| {
                let node = nodes[node_index].clone();
                match node {
                    BVHNode::Node { child_l, child_r, child_l_aabb, child_r_aabb, .. } => {
                        Some((NodeData {
                                  index: child_l,
                                  aabb: child_l_aabb,
                              },
                              NodeData {
                                  index: child_r,
                                  aabb: child_r_aabb,
                              }))
                    }
                    BVHNode::Leaf { .. } => None,
                }
            };

            // Get indices and AABBs of all grandchildren
            let left_children_nodes = {
                let node_data: NodeData = child_l;
                let index = node_data.index;
                get_children_node_data(index)
            };
            let right_children_nodes = {
                let node_data: NodeData = child_r;
                let index = node_data.index;
                get_children_node_data(index)
            };
            (left_children_nodes, right_children_nodes)
        };

        // Consider all the possible rotations, choose the one with the lowest SAH cost
        {
            let mut consider_rotation =
                |a: NodeData, b: NodeData, sa: f32, grandchildren: bool| {
                    if sa < best_surface_area {
                        best_surface_area = sa;
                        best_rotation = Some((a.index, b.index));
                        perform_grandchild_rotation = grandchildren;
                    }
                };

            // Child to grandchild rotations
            if let Some((child_rl, child_rr)) = right_children_nodes {
                consider_rotation(child_l,
                                  child_rl,
                                  child_rl.aabb.surface_area() +
                                  child_l.aabb.join(&child_rr.aabb).surface_area(),
                                  false);
                consider_rotation(child_l,
                                  child_rr,
                                  child_rr.aabb.surface_area() +
                                  child_l.aabb.join(&child_rl.aabb).surface_area(),
                                  false);
            }
            if let Some((child_ll, child_lr)) = left_children_nodes {
                consider_rotation(child_r,
                                  child_ll,
                                  child_ll.aabb.surface_area() +
                                  child_r.aabb.join(&child_lr.aabb).surface_area(),
                                  false);
                consider_rotation(child_r,
                                  child_lr,
                                  child_lr.aabb.surface_area() +
                                  child_r.aabb.join(&child_ll.aabb).surface_area(),
                                  false);

                // Grandchild to grandchild rotations
                if let Some((child_rl, child_rr)) = right_children_nodes {
                    consider_rotation(child_ll,
                                      child_rl,
                                      child_rl.aabb.join(&child_lr.aabb).surface_area() +
                                      child_ll.aabb.join(&child_rr.aabb).surface_area(),
                                      true);
                    consider_rotation(child_ll,
                                      child_rr,
                                      child_ll.aabb.join(&child_rl.aabb).surface_area() +
                                      child_lr.aabb.join(&child_rr.aabb).surface_area(),
                                      true);
                }
            }
        }

        let new_refit_node_index = if parent_index > 0 {
            Some(parent_index)
        } else {
            None
        };

        if let Some(rotation) = best_rotation {
            BVH::rotate(nodes, rotation.0, rotation.1, shapes);

            if perform_grandchild_rotation {
                // Update this node's AABBs (child_l_aabb, child_r_aabb)
                // according to the children nodes' AABBs.

                // The AABBs of the children have changed, so we need to get the new ones.
                // The children are still the same though, so we can use the indices we already have.
                let new_child_l_aabb = BVH::get_node_aabb(&nodes[child_l.index].clone(), shapes);
                let new_child_r_aabb = BVH::get_node_aabb(&nodes[child_r.index].clone(), shapes);

                let mut node = &mut nodes[node_index];
                match node {
                    &mut BVHNode::Node { ref mut child_l_aabb, ref mut child_r_aabb, .. } => {
                        *child_l_aabb = new_child_l_aabb;
                        *child_r_aabb = new_child_r_aabb;
                    }
                    // We know that node must be a BVHNode::Node at this point
                    _ => unreachable!(),
                }
            }

            // Return parent node's index for upcoming refitting,
            // since this node just changed its AABB
            new_refit_node_index
        } else {
            // Update this node's AABBs (child_l_aabb, child_r_aabb)
            // according to the children nodes' AABBs.
            let mut node = &mut nodes[node_index];
            match node {
                &mut BVHNode::Node { ref mut child_l_aabb, ref mut child_r_aabb, .. } => {
                    *child_l_aabb = child_l.aabb;
                    *child_r_aabb = child_r.aabb;
                }
                // We know that node must be a BVHNode::Node at this point
                _ => unreachable!(),
            }

            /// Returns true randomly, with a chance given by the parameter
            fn chance(chance: f32) -> bool {
                let mut rng = thread_rng();
                rng.next_f32() < chance
            }
            // Even with no rotation being useful for this node, a parent node's rotation
            // could be beneficial, so queue the parent *sometimes*.
            // (See https://github.com/jeske/SimpleScene/blob/master/SimpleScene/Util/ssBVH/ssBVH_Node.cs#L307)
            // TODO Evaluate whether or not this is a smart thing to do
            if chance(0.01f32) {
                new_refit_node_index
            } else {
                None
            }
        }
    }

    /// Switch two nodes by rewiring the involved indices (not by moving them in the nodes slice).
    /// Also updates the AABBs of the parents.
    fn rotate<Shape: BHShape>(nodes: &mut Vec<BVHNode>,
                              node_a_index: usize,
                              node_b_index: usize,
                              shapes: &[Shape]) {
        // This function is not defined with a self parameter to make it easier to call
        // without running into borrow checker issues.
        macro_rules! should_not_happen {
            () => ( panic!("While rotating BVH nodes, something unexpected happened."); );
            ($s:expr) => ( panic!("While rotating BVH nodes, something unexpected happened: {}", $s); );
        }

        #[allow(dead_code)] // The compiler falsely detects dead code here
        fn get_parent_index(nodes: &Vec<BVHNode>, node_index: usize) -> usize {
            let node = &nodes[node_index];

            match *node {
                BVHNode::Node { parent, .. } |
                BVHNode::Leaf { parent, .. } => parent,
            }
        }

        // Get parent indices
        let node_a_parent_index = get_parent_index(nodes, node_a_index);
        let node_b_parent_index = get_parent_index(nodes, node_b_index);

        #[allow(dead_code)] // The compiler falsely detects dead code here
        fn get_is_left_child(nodes: &Vec<BVHNode>,
                             node_index: usize,
                             node_parent_index: usize)
                             -> bool {
            let node_parent = &nodes[node_parent_index];

            match *node_parent {
                BVHNode::Node { child_l, .. } => child_l == node_index,
                _ => should_not_happen!(),
            }
        }

        // Get info about the nodes being a left or right child
        let node_a_is_left_child = get_is_left_child(nodes, node_a_index, node_a_parent_index);
        let node_b_is_left_child = get_is_left_child(nodes, node_b_index, node_b_parent_index);

        #[allow(dead_code)] // The compiler falsely detects dead code here
        fn connect_nodes<Shape: BHShape>(nodes: &mut Vec<BVHNode>,
                                         child_index: usize,
                                         parent_index: usize,
                                         left_child: bool,
                                         shapes: &[Shape]) {
            let child_aabb = BVH::get_node_aabb(&nodes[child_index].clone(), shapes);

            // Set parent's child and child_aabb; and get its depth
            let parent_depth = {
                let mut parent = &mut nodes[parent_index];
                match parent {
                    &mut BVHNode::Node { ref mut child_l,
                                         ref mut child_r,
                                         ref mut child_l_aabb,
                                         ref mut child_r_aabb,
                                         depth,
                                         .. } => {
                        if left_child {
                            *child_l = child_index;
                            *child_l_aabb = child_aabb;
                        } else {
                            *child_r = child_index;
                            *child_r_aabb = child_aabb;
                        }
                        depth
                    }
                    _ => should_not_happen!(),
                }
            };

            // Set child's parent and depth
            {
                let mut child = &mut nodes[child_index];
                match child {
                    &mut BVHNode::Node { ref mut parent, ref mut depth, .. } |
                    &mut BVHNode::Leaf { ref mut parent, ref mut depth, .. } => {
                        *parent = parent_index;
                        *depth = parent_depth + 1;
                    }
                };
            }
        }

        // Perform the switch
        connect_nodes(nodes,
                      node_a_index,
                      node_b_parent_index,
                      node_b_is_left_child,
                      shapes);
        connect_nodes(nodes,
                      node_b_index,
                      node_a_parent_index,
                      node_a_is_left_child,
                      shapes);
    }

    /// Gets the `AABB` for a `BVHNode`.
    /// Returns the shape's `AABB` for leaves, and the joined `AABB` of
    /// the two children's `AABB`s for non-leaves.
    fn get_node_aabb<Shape: BHShape>(node: &BVHNode, shapes: &[Shape]) -> AABB {
        match *node {
            BVHNode::Node { child_l_aabb, child_r_aabb, .. } => child_l_aabb.join(&child_r_aabb),
            BVHNode::Leaf { shape, .. } => shapes[shape].aabb(),
        }
    }
}

#[cfg(test)]
pub mod tests {
    use bvh::{BVH, BVHNode};
    use std::collections::HashSet;
    use testbase::build_some_bh;

    impl PartialEq for BVHNode {
        // TODO Consider also comparing AABBs
        fn eq(&self, other: &BVHNode) -> bool {
            match *self {
                BVHNode::Node { parent, depth, child_l, child_r, .. } => {
                    let (self_parent, self_depth, self_child_l, self_child_r) =
                        (parent, depth, child_l, child_r);
                    if let BVHNode::Node { parent, depth, child_l, child_r, .. } = *other {
                        (self_parent, self_depth, self_child_l, self_child_r) ==
                        (parent, depth, child_l, child_r)
                    } else {
                        false
                    }
                }
                BVHNode::Leaf { parent, depth, shape } => {
                    let (self_parent, self_depth, self_shape) = (parent, depth, shape);
                    if let BVHNode::Leaf { parent, depth, shape } = *other {
                        (self_parent, self_depth, self_shape) == (parent, depth, shape)
                    } else {
                        false
                    }
                }
            }
        }
    }

    #[test]
    /// Tests if the optimize function tries to change a fresh BVH even though it shouldn't
    fn test_optimizing_new_bvh() {
        let (mut shapes, mut bvh) = build_some_bh::<BVH>();

        let original_nodes = bvh.nodes.clone();

        let refit_shape_indices: HashSet<usize> = (0..shapes.len()).collect();
        bvh.optimize(&refit_shape_indices, &mut shapes);

        let optimized_nodes = bvh.nodes.clone();
        for i in 0..optimized_nodes.len() {
            assert_eq!(optimized_nodes[i], original_nodes[i]);
        }
    }

    // TODO Add tests for:
    // * Correct (as in error-free) BVH after optimizing
    // * Compare a BVH with updated shapes with a fresh BVH made with the same shapes:
    //   * A BVH with two near and one far shape. Move one of the two close to the far one.

    // TODO Add benchmarks for:
    // * Optimizing an optimal bvh
    // * Optimizing a bvh after randomizing 50% of the shapes
    // * Optimizing a bvh after randomizing all shapes (to compare with a full rebuild)
}
