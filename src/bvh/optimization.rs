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

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
enum OptimizationIndex {
    Refit(usize),
    FixAABBs(usize),
}

impl BVHNode {
    fn parent(&self) -> usize {
        match self {
            &BVHNode::Node { parent, .. } |
            &BVHNode::Leaf { parent, .. } => parent,
        }
    }

    fn depth(&self) -> u32 {
        match self {
            &BVHNode::Node { depth, .. } |
            &BVHNode::Leaf { depth, .. } => depth,
        }
    }

    fn child_l(&self) -> usize {
        match self {
            &BVHNode::Node { child_l, .. } => child_l,
            _ => panic!(),
        }
    }

    fn child_r(&self) -> usize {
        match self {
            &BVHNode::Node { child_r, .. } => child_r,
            _ => panic!(),
        }
    }

    fn child_l_aabb(&self) -> AABB {
        match self {
            &BVHNode::Node { child_l_aabb, .. } => child_l_aabb,
            _ => panic!(),
        }
    }

    fn child_r_aabb(&self) -> AABB {
        match self {
            &BVHNode::Node { child_r_aabb, .. } => child_r_aabb,
            _ => panic!(),
        }
    }

    /// Gets the `AABB` for a `BVHNode`.
    /// Returns the shape's `AABB` for leaves, and the joined `AABB` of
    /// the two children's `AABB`s for non-leaves.
    fn get_node_aabb<Shape: BHShape>(&self, shapes: &[Shape]) -> AABB {
        match self {
            &BVHNode::Node { child_l_aabb, child_r_aabb, .. } => child_l_aabb.join(&child_r_aabb),
            &BVHNode::Leaf { shape, .. } => shapes[shape].aabb(),
        }
    }
}

impl BVH {
    /// Optimizes the `BVH` by batch-reorganizing updated nodes.
    /// Based on https://github.com/jeske/SimpleScene/blob/master/SimpleScene/Util/ssBVH/ssBVH.cs
    ///
    /// Needs all the scene's shapes, plus the indices of the shapes that were updated.
    ///
    pub fn optimize<Shape: BHShape>(&mut self,
                                    refit_shape_indices: &HashSet<usize>,
                                    shapes: &[Shape]) {
        let mut refit_node_indices: HashSet<OptimizationIndex> = refit_shape_indices.iter()
            .map(|x| OptimizationIndex::Refit(shapes[*x].bh_node_index()))
            .collect();

        println!("{} nodes for refitting.", refit_node_indices.len());

        // As long as we have refit nodes left, take the list of refit nodes
        // with the highest depth (sweep nodes) and try to rotate them all
        while refit_node_indices.len() > 0 {
            let mut max_depth = 0;
            let mut sweep_node_indices = Vec::new();

            // TODO Find sweep nodes by sorting the list once (high-depth-first) before starting
            // Find max_depth and sweep_node_indices in one iteration
            for refit_node_index in refit_node_indices.iter() {
                let index = match refit_node_index {
                    &OptimizationIndex::Refit(index) |
                    &OptimizationIndex::FixAABBs(index) => index,
                };

                let depth = match self.nodes[index] {
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

            println!("{} sweep nodes, depth {}.",
                     sweep_node_indices.len(),
                     max_depth);

            // Try to find a useful tree rotation with all nodes previously found
            for sweep_node_index in &sweep_node_indices {
                // This node does not need to be checked again
                refit_node_indices.remove(sweep_node_index);

                // TODO There might be multithreading potential here
                let new_refit_node_index = match sweep_node_index {
                    &OptimizationIndex::Refit(index) => self.try_rotate(index, shapes),
                    &OptimizationIndex::FixAABBs(index) => {
                        BVH::fix_aabbs(&mut self.nodes, index, shapes)
                    }
                };

                // Instead of finding a useful tree rotation, we found another node
                // that we should check, so we add its index to the refit_node_indices.
                if let Some(index) = new_refit_node_index {
                    // TODO insert to the appropriate end of the list
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
    fn try_rotate<Shape: BHShape>(&mut self,
                                  node_index: usize,
                                  shapes: &[Shape])
                                  -> Option<OptimizationIndex> {
        print!("   [{}]\t", node_index);
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
            BVHNode::Node { parent, child_l, child_r, .. } => {
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
                                println!("Setting {:?} from {}", child_l_aabb, child_l);
                                println!("\tand {:?} from {}.", child_r_aabb, child_r);
                            }
                            // We know that node must be a BVHNode::Node at this point
                            _ => unreachable!(),
                        }

                        return Some(OptimizationIndex::Refit(parent));
                    }
                }

                // Recalculate AABBs for the children since at least one of them changed.
                // Don't update the AABBs in the node yet because they're still subject to change
                // during potential upcoming rotations.
                let aabb_l = nodes[child_l].get_node_aabb(shapes);
                let aabb_r = nodes[child_r].get_node_aabb(shapes);

                parent_index = parent;
                best_surface_area = aabb_l.surface_area() + aabb_r.surface_area();
                (NodeData {
                     index: child_l,
                     aabb: aabb_l,
                 },
                 NodeData {
                     index: child_r,
                     aabb: aabb_r,
                 })
            }
            BVHNode::Leaf { parent, shape, .. } => {
                println!("IAmA leaf node. Queueing parent ({}). {:?}.",
                         parent,
                         shapes[shape].aabb());
                return Some(OptimizationIndex::Refit(parent));
            }
        };

        match node_clone {
            BVHNode::Node { .. } => println!("Trying to rotate."),
            _ => unreachable!(),
        }

        // Stores the Rotation that would result in the surface area best_surface_area,
        // thus being the favored rotation that will be executed after considering all rotations.
        let mut best_rotation: Option<(usize, usize)> = None;

        // Get the grandchildren's NodeData
        let (left_children_nodes, right_children_nodes) = {
            let get_children_node_data = |node_index: usize| {
                let node = &nodes[node_index];
                match node {
                    &BVHNode::Node { child_l, child_r, child_l_aabb, child_r_aabb, .. } => {
                        Some((NodeData {
                                  index: child_l,
                                  aabb: child_l_aabb,
                              },
                              NodeData {
                                  index: child_r,
                                  aabb: child_r_aabb,
                              }))
                    }
                    &BVHNode::Leaf { .. } => None,
                }
            };

            // Get indices and AABBs of all grandchildren
            let left_children_nodes = get_children_node_data(child_l.index);
            let right_children_nodes = get_children_node_data(child_r.index);
            (left_children_nodes, right_children_nodes)
        };

        // Consider all the possible rotations, choose the one with the lowest SAH cost
        {
            let mut consider_rotation = |a: NodeData, b: NodeData, sa: f32| {
                println!("        SA {} vs. current SA {}.", sa, best_surface_area);
                if sa < best_surface_area {
                    best_surface_area = sa;
                    best_rotation = Some((a.index, b.index));
                }
            };

            // Child to grandchild rotations
            if let Some((child_rl, child_rr)) = right_children_nodes {
                {
                    let sa = child_rl.aabb.surface_area() +
                             child_l.aabb.join(&child_rr.aabb).surface_area();
                    consider_rotation(child_l, child_rl, sa);
                }
                {
                    let sa = child_rr.aabb.surface_area() +
                             child_l.aabb.join(&child_rl.aabb).surface_area();
                    consider_rotation(child_l, child_rr, sa);
                }
            }
            if let Some((child_ll, child_lr)) = left_children_nodes {
                {
                    let sa = child_ll.aabb.surface_area() +
                             child_r.aabb.join(&child_lr.aabb).surface_area();
                    consider_rotation(child_r, child_ll, sa);
                }
                {
                    let sa = child_lr.aabb.surface_area() +
                             child_r.aabb.join(&child_ll.aabb).surface_area();
                    consider_rotation(child_r, child_lr, sa);
                }

                // Grandchild to grandchild rotations
                if let Some((child_rl, child_rr)) = right_children_nodes {
                    {
                        let sa = child_rl.aabb.join(&child_lr.aabb).surface_area() +
                                 child_ll.aabb.join(&child_rr.aabb).surface_area();
                        consider_rotation(child_ll, child_rl, sa);
                    }
                    {
                        let sa = child_ll.aabb.join(&child_rl.aabb).surface_area() +
                                 child_lr.aabb.join(&child_rr.aabb).surface_area();
                        consider_rotation(child_ll, child_rr, sa);
                    }
                }
            }
        }

        if let Some((rotation_node_a, rotation_node_b)) = best_rotation {
            BVH::rotate(nodes, rotation_node_a, rotation_node_b, shapes);

            // Update the children's children AABBs and the children AABBs of node
            BVH::fix_children_and_own_aabbs(nodes, node_index, shapes);

            // Return parent node's index for upcoming refitting,
            // since this node just changed its AABB
            if node_index > 0 {
                Some(OptimizationIndex::Refit(parent_index))
            } else {
                None
            }
        } else {
            println!("    No useful rotation.");
            // Update this node's AABBs (child_l_aabb, child_r_aabb)
            // according to the children nodes' AABBs.
            BVH::set_aabbs(nodes, node_index, child_l.aabb, child_r.aabb);

            /// Returns true randomly, with a chance given by the parameter
            fn chance(chance: f32) -> bool {
                let mut rng = thread_rng();
                rng.next_f32() < chance
            }

            if node_index > 0 {
                // Even with no rotation being useful for this node, a parent node's rotation
                // could be beneficial, so queue the parent *sometimes*.
                // (See https://github.com/jeske/SimpleScene/blob/master/SimpleScene/Util/ssBVH/ssBVH_Node.cs#L307)
                // TODO Evaluate whether or not this is a smart thing to do
                // if chance(0.01f32) {
                Some(OptimizationIndex::Refit(parent_index))
                // } else {
                //     Some(OptimizationIndex::FixAABBs(parent_index))
                // }
            } else {
                None
            }
        }
    }

    /// Sets child_l_aabb and child_r_aabb of a BVHNode::Node to match its children,
    /// right after updating the children themselves. Not recursive.
    fn fix_children_and_own_aabbs<Shape: BHShape>(nodes: &mut Vec<BVHNode>,
                                                  node_index: usize,
                                                  shapes: &[Shape]) {
        let (child_l, child_r) = {
            let node = &nodes[node_index];
            match node {
                &BVHNode::Node { child_l, child_r, .. } => (child_l, child_r),
                // node must be a BVHNode::Node for this function
                _ => panic!(),
            }
        };

        BVH::fix_aabbs(nodes, child_l, shapes);
        BVH::fix_aabbs(nodes, child_r, shapes);

        let child_l_aabb = nodes[child_l].get_node_aabb(shapes);
        let child_r_aabb = nodes[child_r].get_node_aabb(shapes);

        BVH::set_aabbs(nodes, node_index, child_l_aabb, child_r_aabb);
    }

    /// Sets child_l_aabb and child_r_aabb of a BVHNode::Node to match its children.
    fn fix_aabbs<Shape: BHShape>(nodes: &mut Vec<BVHNode>,
                                 node_index: usize,
                                 shapes: &[Shape])
                                 -> Option<OptimizationIndex> {
        let (child_l, child_r, parent) = {
            let node = &nodes[node_index];
            match node {
                &BVHNode::Node { child_l, child_r, parent, .. } => (child_l, child_r, parent),
                // Don't do anything if the node is a leaf
                _ => return None,
            }
        };

        let child_l_aabb = nodes[child_l].get_node_aabb(shapes);
        let child_r_aabb = nodes[child_r].get_node_aabb(shapes);

        BVH::set_aabbs(nodes, node_index, child_l_aabb, child_r_aabb);

        Some(OptimizationIndex::FixAABBs(parent))
    }

    /// Sets the children AABBs of a BVHNode::Node.
    fn set_aabbs(nodes: &mut Vec<BVHNode>,
                 node_index: usize,
                 new_child_l_aabb: AABB,
                 new_child_r_aabb: AABB) {
        let node = &mut nodes[node_index];
        match node {
            &mut BVHNode::Node { ref mut child_l_aabb, ref mut child_r_aabb, .. } => {
                *child_l_aabb = new_child_l_aabb;
                *child_r_aabb = new_child_r_aabb;
            }
            // node must be a BVHNode::Node for this function
            _ => panic!(),
        }
    }

    /// Switch two nodes by rewiring the involved indices (not by moving them in the nodes slice).
    /// Also updates the AABBs of the parents.
    fn rotate<Shape: BHShape>(nodes: &mut Vec<BVHNode>,
                              node_a_index: usize,
                              node_b_index: usize,
                              shapes: &[Shape]) {
        println!("    ROTATING {} and {}", node_a_index, node_b_index);

        // Get parent indices
        let node_a_parent_index = nodes[node_a_index].parent();
        let node_b_parent_index = nodes[node_b_index].parent();

        // Get info about the nodes being a left or right child
        let node_a_is_left_child = BVH::node_is_left_child(nodes, node_a_index);
        let node_b_is_left_child = BVH::node_is_left_child(nodes, node_b_index);

        // Perform the switch
        BVH::connect_nodes(nodes,
                           node_a_index,
                           node_b_parent_index,
                           node_b_is_left_child,
                           shapes);
        BVH::connect_nodes(nodes,
                           node_b_index,
                           node_a_parent_index,
                           node_a_is_left_child,
                           shapes);
    }

    /// Updates the depth of a node, and sets the depth of its descendants accordingly
    fn update_depth_recursively(nodes: &mut Vec<BVHNode>, node_index: usize, new_depth: u32) {
        let children = {
            let node = &mut nodes[node_index];
            match node {
                &mut BVHNode::Node { ref mut depth, child_l, child_r, .. } => {
                    *depth = new_depth;
                    Some((child_l, child_r))
                }
                &mut BVHNode::Leaf { ref mut depth, .. } => {
                    *depth = new_depth;
                    None
                }
            }
        };
        if let Some((child_l, child_r)) = children {
            BVH::update_depth_recursively(nodes, child_l, new_depth + 1);
            BVH::update_depth_recursively(nodes, child_r, new_depth + 1);
        }
    }

    fn node_is_left_child(nodes: &Vec<BVHNode>, node_index: usize) -> bool {
        let node = &nodes[node_index];
        let node_parent_index = node.parent();

        let node_parent = &nodes[node_parent_index];

        match node_parent {
            &BVHNode::Node { child_l, .. } => child_l == node_index,
            // Assuming that our BVH is correct, the parent cannot be a leaf
            _ => unreachable!(),
        }
    }

    fn connect_nodes<Shape: BHShape>(nodes: &mut Vec<BVHNode>,
                                     child_index: usize,
                                     parent_index: usize,
                                     left_child: bool,
                                     shapes: &[Shape]) {
        let child_aabb = nodes[child_index].get_node_aabb(shapes);
        println!("\tConnecting: {} < {}.", child_index, parent_index);
        // Set parent's child and child_aabb; and get its depth
        let parent_depth = {
            let parent = &mut nodes[parent_index];
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
                    println!("\t  {}'s new {:?}", parent_index, child_aabb);
                    depth
                }
                // Assuming that our BVH is correct, the parent cannot be a leaf
                _ => unreachable!(),
            }
        };

        // Set child's parent
        {
            let child = &mut nodes[child_index];
            match child {
                &mut BVHNode::Node { ref mut parent, .. } |
                &mut BVHNode::Leaf { ref mut parent, .. } => {
                    *parent = parent_index;
                }
            };
        }

        // Update the node's and the node's descendants' depth values
        BVH::update_depth_recursively(nodes, child_index, parent_depth + 1);
    }
}

#[cfg(test)]
pub mod tests {
    use EPSILON;
    use bvh::{BVH, BVHNode};
    use aabb::{AABB, Bounded};
    use std::collections::HashSet;
    use std::rand::{task_rng, Rng};
    use nalgebra::{ApproxEq, Point3};
    use testbase::{build_some_bh, UnitBox};
    use bounding_hierarchy::BHShape;
    use std::f32;

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

    fn aabb_is_in_aabb(outer: &AABB, inner: &AABB) -> bool {
        println!("Checking whether");
        println!("    {:?}", inner);
        println!("  is in");
        println!("    {:?}.", outer);
        println!("");
        outer.approx_contains_eps(&inner.min, EPSILON) &&
        outer.approx_contains_eps(&inner.max, EPSILON)
    }

    fn assert_aabb_approx_eq(a: &AABB, b: &AABB) {
        assert_approx_eq_eps!(a.min, b.min, EPSILON);
        assert_approx_eq_eps!(a.max, b.max, EPSILON);
    }

    /// Checks if all children of a node have the correct parent index,
    /// and that there is no detached subtree.
    // TODO Consider moving into testbase and use more often than after optimization
    fn assert_correct_bvh(bvh: &BVH) {
        let nodes = &bvh.nodes;
        // The counter for all nodes that are (grand)children of the root node
        let mut node_count = 0usize;

        fn assert_correct_subtree(nodes: &Vec<BVHNode>,
                                  index: usize,
                                  parent_index: usize,
                                  outer_aabb: &AABB,
                                  expected_depth: u32,
                                  node_count: &mut usize) {
            let node_clone = &nodes[index];
            *node_count += 1;
            match node_clone {
                &BVHNode::Node { parent, depth, child_l, child_r, child_l_aabb, child_r_aabb } => {
                    assert_eq!(parent, parent_index, "Wrong parent index.");
                    assert_eq!(expected_depth, depth, "Wrong depth.");
                    print!("[{}<{}] ", child_l, index);
                    assert!(aabb_is_in_aabb(outer_aabb, &child_l_aabb));
                    print!("[{}<{}] ", child_r, index);
                    assert!(aabb_is_in_aabb(outer_aabb, &child_r_aabb));
                    assert_correct_subtree(nodes,
                                           child_l,
                                           index,
                                           &child_l_aabb,
                                           expected_depth + 1,
                                           node_count);
                    assert_correct_subtree(nodes,
                                           child_r,
                                           index,
                                           &child_r_aabb,
                                           expected_depth + 1,
                                           node_count);
                }
                &BVHNode::Leaf { parent, depth, .. } => {
                    assert_eq!(parent, parent_index, "Wrong parent index (leaf).");
                    assert_eq!(expected_depth, depth, "Wrong depth (leaf).");
                }
            }
        }

        let space = AABB {
            min: Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
            max: Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
        };

        assert_correct_subtree(nodes, 0, 0, &space, 0, &mut node_count);

        // Check if all nodes have been counted from the root node.
        // If this assert fails, it means we have a detached subtree.
        assert_eq!(node_count, nodes.len(), "Detached subtree.");
    }

    fn update_unit_boxes(shapes: &mut Vec<UnitBox>) -> HashSet<usize> {
        shapes[0].pos = Point3::new(10.0, 1.0, 2.0);
        shapes[1].pos = Point3::new(-10.0, -10.0, 10.0);
        shapes[2].pos = Point3::new(-10.0, 10.0, 10.0);
        shapes[3].pos = Point3::new(-10.0, 10.0, -10.0);
        shapes[4].pos = Point3::new(11.0, 1.0, 2.0);
        shapes[5].pos = Point3::new(11.0, 2.0, 2.0);

        (0..6).collect()
    }

    #[test]
    /// Tests if the optimize function tries to change a fresh BVH even though it shouldn't
    fn test_optimizing_new_bvh() {
        let (mut shapes, mut bvh) = build_some_bh::<BVH>();

        let original_nodes = bvh.nodes.clone();

        let refit_shape_indices: HashSet<usize> = (0..shapes.len()).collect();
        // let refit_shape_indices = update_unit_boxes(&mut shapes);
        bvh.optimize(&refit_shape_indices, &shapes);

        for (optimized, original) in bvh.nodes.iter().zip(original_nodes.iter()) {
            assert_eq!(optimized, original);
        }
    }

    #[test]
    /// Tests whether or not a BVH is still correct (as in error-free)
    /// after a few optimization calls
    fn test_correctness_after_optimize() {
        let (mut shapes, mut bvh) = build_some_bh::<BVH>();

        let refit_shape_indices = update_unit_boxes(&mut shapes);
        bvh.optimize(&refit_shape_indices, &shapes);

        assert_correct_bvh(&bvh);
    }

    #[test]
    /// Test whether a simple update on a simple BVH yields the expected optimization result
    fn test_optimize_simple_update() {
        let mut shapes = Vec::new();

        shapes.push(UnitBox::new(0, Point3::new(-50.0, 0.0, 0.0)));
        shapes.push(UnitBox::new(1, Point3::new(-40.0, 0.0, 0.0)));
        shapes.push(UnitBox::new(2, Point3::new(50.0, 0.0, 0.0)));

        let mut bvh = BVH::build(&mut shapes);

        bvh.pretty_print();

        {
            let left = &shapes[0];
            let moving = &shapes[1];

            let nodes = &bvh.nodes;

            match nodes[left.bh_node_index()] {
                BVHNode::Leaf { parent, .. } => {
                    let left_parent = parent;

                    match nodes[moving.bh_node_index()] {
                        BVHNode::Leaf { parent, .. } => {
                            assert_eq!(parent, left_parent);
                        }
                        _ => panic!(),
                    }
                }
                _ => panic!(),
            };
        }

        {
            let mut moving = &mut shapes[1];
            moving.pos = Point3::new(40.0, 0.0, 0.0);
        }

        let mut refit_shape_indices: HashSet<usize> = (1..2).collect();
        bvh.optimize(&refit_shape_indices, &shapes);

        bvh.pretty_print();

        assert_correct_bvh(&bvh);

        {
            let moving = &shapes[1];
            let right = &shapes[2];

            let nodes = &bvh.nodes;

            match nodes[right.bh_node_index()] {
                BVHNode::Leaf { parent, .. } => {
                    let right_parent = parent;

                    match nodes[moving.bh_node_index()] {
                        BVHNode::Leaf { parent, .. } => {
                            assert_eq!(parent, right_parent);
                        }
                        _ => panic!(),
                    }
                }
                _ => panic!(),
            }
        }
    }

    fn create_predictable_bvh() -> (Vec<UnitBox>, Vec<BVHNode>) {
        let mut shapes = Vec::new();
        shapes.push(UnitBox::new(0, Point3::new(0.0, 0.0, 0.0)));
        shapes.push(UnitBox::new(1, Point3::new(2.0, 0.0, 0.0)));
        shapes.push(UnitBox::new(2, Point3::new(4.0, 0.0, 0.0)));
        shapes.push(UnitBox::new(3, Point3::new(6.0, 0.0, 0.0)));

        let mut nodes = Vec::new();

        // Root node
        nodes.push(BVHNode::Node {
            parent: 0,
            depth: 0,
            child_l_aabb: shapes[0].aabb().join(&shapes[1].aabb()),
            child_l: 1,
            child_r_aabb: shapes[2].aabb().join(&shapes[3].aabb()),
            child_r: 2,
        });

        // Depth 1 nodes
        nodes.push(BVHNode::Node {
            parent: 0,
            depth: 1,
            child_l_aabb: shapes[0].aabb(),
            child_l: 3,
            child_r_aabb: shapes[1].aabb(),
            child_r: 4,
        });
        nodes.push(BVHNode::Node {
            parent: 0,
            depth: 1,
            child_l_aabb: shapes[2].aabb(),
            child_l: 5,
            child_r_aabb: shapes[3].aabb(),
            child_r: 6,
        });

        // Depth 2 nodes (leaves)
        nodes.push(BVHNode::Leaf {
            parent: 1,
            depth: 2,
            shape: 0,
        });
        nodes.push(BVHNode::Leaf {
            parent: 1,
            depth: 2,
            shape: 1,
        });
        nodes.push(BVHNode::Leaf {
            parent: 2,
            depth: 2,
            shape: 2,
        });
        nodes.push(BVHNode::Leaf {
            parent: 2,
            depth: 2,
            shape: 3,
        });

        (shapes, nodes)
    }

    #[test]
    fn test_connect_grandchildren() {
        let (shapes, mut nodes) = create_predictable_bvh();

        // Switch two nodes
        BVH::connect_nodes(&mut nodes, 3, 2, true, &shapes);
        BVH::connect_nodes(&mut nodes, 5, 1, true, &shapes);

        // Check if the resulting tree is as expected
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

        assert_aabb_approx_eq(&nodes[1].child_l_aabb(), &shapes[2].aabb());
        assert_aabb_approx_eq(&nodes[1].child_r_aabb(), &shapes[1].aabb());
        assert_aabb_approx_eq(&nodes[2].child_l_aabb(), &shapes[0].aabb());
        assert_aabb_approx_eq(&nodes[2].child_r_aabb(), &shapes[3].aabb());
    }

    #[test]
    fn test_connect_child_grandchild() {
        let (shapes, mut nodes) = create_predictable_bvh();

        // Switch two nodes
        BVH::connect_nodes(&mut nodes, 1, 2, true, &shapes);
        BVH::connect_nodes(&mut nodes, 5, 0, true, &shapes);

        // Check if the resulting tree is as expected
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

        assert_aabb_approx_eq(&nodes[0].child_l_aabb(), &shapes[2].aabb());

        assert_aabb_approx_eq(&nodes[2].child_r_aabb(), &shapes[3].aabb());

        assert_aabb_approx_eq(&nodes[1].child_l_aabb(), &shapes[0].aabb());
        assert_aabb_approx_eq(&nodes[1].child_r_aabb(), &shapes[1].aabb());
    }

    #[test]
    fn test_rotate_grandchildren() {
        let (shapes, mut nodes) = create_predictable_bvh();

        // Switch two nodes
        BVH::rotate(&mut nodes, 3, 5, &shapes);

        // Check if the resulting tree is as expected
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

        assert_aabb_approx_eq(&nodes[1].child_l_aabb(), &shapes[2].aabb());
        assert_aabb_approx_eq(&nodes[1].child_r_aabb(), &shapes[1].aabb());
        assert_aabb_approx_eq(&nodes[2].child_l_aabb(), &shapes[0].aabb());
        assert_aabb_approx_eq(&nodes[2].child_r_aabb(), &shapes[3].aabb());
    }

    #[test]
    fn test_rotate_child_grandchild() {
        let (shapes, mut nodes) = create_predictable_bvh();

        // Switch two nodes
        BVH::rotate(&mut nodes, 1, 5, &shapes);

        // Check if the resulting tree is as expected
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

        assert_aabb_approx_eq(&nodes[0].child_l_aabb(), &shapes[2].aabb());

        assert_aabb_approx_eq(&nodes[2].child_r_aabb(), &shapes[3].aabb());

        assert_aabb_approx_eq(&nodes[1].child_l_aabb(), &shapes[0].aabb());
        assert_aabb_approx_eq(&nodes[1].child_r_aabb(), &shapes[1].aabb());
    }

    #[test]
    fn test_try_rotate_child_grandchild() {
        let (mut shapes, mut nodes) = create_predictable_bvh();

        shapes[2].pos = Point3::new(-40.0, 0.0, 0.0);

        let nodes = {
            let mut bvh = BVH { nodes: nodes };
            bvh.try_rotate(2, &shapes); // Try to rotate node 2 because node 5 changed
            bvh.try_rotate(0, &shapes); // Try to rotate node 0 because rotating node 2 should not have yielded a result
            bvh.nodes
        };

        // Check if the resulting tree is as expected
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

        assert_aabb_approx_eq(&nodes[0].child_l_aabb(), &shapes[2].aabb());
        let right_subtree_aabb = &shapes[0].aabb().join(&shapes[1].aabb()).join(&shapes[3].aabb());
        assert_aabb_approx_eq(&nodes[0].child_r_aabb(), right_subtree_aabb);

        assert_aabb_approx_eq(&nodes[2].child_r_aabb(), &shapes[3].aabb());

        assert_aabb_approx_eq(&nodes[1].child_l_aabb(), &shapes[0].aabb());
        assert_aabb_approx_eq(&nodes[1].child_r_aabb(), &shapes[1].aabb());
    }

    fn randomly_move_triangles(triangles: &mut Vec<Triangle>, amount: usize, seed: &mut u64) -> HashSet<usize> {
        let mut indices : Vec<usize> = (0..triangles.len()).collect();
        thread_rng().shuffle(&mut slice);
        indices.truncate(amount);

        for index in &indices {
            let random_pos = next_point3(seed);
            let b = triangles[index].b - triangles[index].a;
            let c = triangles[index].b - triangles[index].a;
            triangles[index] = Triangle::new(random_pos, random_pos + b, random_pos + c);
        }

        indices.into_iter().collect()
    }

    #[bench]
    /// Benchmark optimizing a bvh after randomizing 50% of the shapes
    fn bench_optimize_optimal_bvh_120k_triangles(b: &mut ::test::Bencher) {
        let mut triangles = create_n_cubes(10_000);
        let mut structure = BVH::build(&mut triangles);
        let mut seed = 0;

        b.iter(|| {
            let updated = randomly_move_triangles(&mut triangles, 60_000, &mut seed);
            structure.optimize(&updated, &triangles);
        });
    }
    // TODO Add benchmarks for:
    // * Benchmark optimizing an optimal bvh
    // * Optimizing a bvh after randomizing all shapes (to compare with a full rebuild)
}
