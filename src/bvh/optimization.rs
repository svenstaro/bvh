//! This module defines the optimization function for the [`Bvh`].
//! By passing the indices of shapes that have changed, the function determines possible
//! tree rotations and optimizes the Bvh using a SAH.
//! Based on [`http://www.cs.utah.edu/~thiago/papers/rotations.pdf`]
//!
//! [`Bvh`]: struct.Bvh.html
//!

use crate::aabb::Aabb;
use crate::bounding_hierarchy::BHShape;
use crate::bvh::*;

use log::info;
use nalgebra::{ClosedAdd, ClosedMul, ClosedSub, Scalar, SimdPartialOrd, ClosedDiv};
use num::{FromPrimitive, Zero, Signed};
use rand::{thread_rng, Rng};
use std::collections::HashSet;

// TODO Consider: Instead of getting the scene's shapes passed, let leaf nodes store an `Aabb`
// that is updated from the outside, perhaps by passing not only the indices of the changed
// shapes, but also their new `Aabb`'s into optimize().
// TODO Consider: Stop updating `Aabb`'s upwards the tree once an `Aabb` didn't get changed.

#[derive(Eq, PartialEq, Hash, Copy, Clone, Debug)]
enum OptimizationIndex {
    Refit(usize),
    FixAabbs(usize),
}

impl OptimizationIndex {
    fn index(&self) -> usize {
        match *self {
            OptimizationIndex::Refit(index) | OptimizationIndex::FixAabbs(index) => index,
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct NodeData<T: Scalar + Copy, const D: usize> {
    index: usize,
    aabb: Aabb<T, D>,
}

impl<T: Scalar + Copy, const D: usize> BvhNode<T, D> {
    // Get the grandchildren's NodeData.
    fn get_children_node_data(&self) -> Option<(NodeData<T, D>, NodeData<T, D>)> {
        match *self {
            BvhNode::Node {
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
            BvhNode::Leaf { .. } => None,
        }
    }
}

impl<T, const D: usize> Bvh<T, D>
where
    T: Scalar
        + Copy
        + FromPrimitive
        + ClosedSub
        + ClosedMul
        + ClosedAdd
        + ClosedDiv
        + Zero
        + SimdPartialOrd
        + PartialOrd
        + Signed
        + std::fmt::Display,
{
    /// This method is called for each node which has been modified and needs to be updated.
    /// If the specified node is a grandparent, then try to optimize the [`Bvh`] by rotating its
    /// children.
    fn update<Shape: BHShape<T, D>>(
        &mut self,
        node_index: usize,
        shapes: &[Shape],
    ) -> Option<OptimizationIndex> {
        info!("   [{}]\t", node_index);

        match self.nodes[node_index] {
            BvhNode::Leaf {
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
            BvhNode::Node {
                parent_index,
                child_l_index,
                child_r_index,
                ..
            } => {
                // The current node is a parent.
                if let (
                    &BvhNode::Leaf {
                        shape_index: shape_l_index,
                        ..
                    },
                    &BvhNode::Leaf {
                        shape_index: shape_r_index,
                        ..
                    },
                ) = (&self.nodes[child_l_index], &self.nodes[child_r_index])
                {
                    // The current node is a final parent. Update its `Aabb`s, because at least
                    // one of its children was updated and queue its parent for refitting.
                    if let BvhNode::Node {
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
        child_l_aabb: &Aabb<T, D>,
        child_r_index: usize,
        child_r_aabb: &Aabb<T, D>,
    ) -> Option<(usize, usize)> {
        // Get indices and `Aabb`s of all grandchildren.
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
            let mut consider_rotation = |new_rotation: (usize, usize), surface_area: T| {
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
    /// the given node (specified by `node_index`) that would improve the [`Bvh`].
    /// If there is, the best rotation found is performed.
    ///
    /// # Preconditions
    ///
    /// This function requires that the subtree at `node_index` has correct [`Aabb`]s.
    ///
    /// # Returns
    ///
    /// `Some(index_of_node)` if a new node was found that should be used for optimization.
    ///
    fn try_rotate<Shape: BHShape<T, D>>(
        &mut self,
        node_index: usize,
        shapes: &[Shape],
    ) -> Option<OptimizationIndex> {
        let (parent_index, child_l_index, child_r_index) = if let BvhNode::Node {
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

        // Recalculate `Aabb`s for the children since at least one of them changed.  Don't update
        // the `Aabb`s in the node yet because they're still subject to change during potential
        // upcoming rotations.
        let child_l_aabb = self.nodes[child_l_index].get_node_aabb(shapes);
        let child_r_aabb = self.nodes[child_r_index].get_node_aabb(shapes);

        let best_rotation =
            self.find_better_rotation(child_l_index, &child_l_aabb, child_r_index, &child_r_aabb);

        if let Some((rotation_node_a, rotation_node_b)) = best_rotation {
            self.rotate(rotation_node_a, rotation_node_b, shapes);

            // Update the children's children `Aabb`s and the children `Aabb`s of node.
            self.fix_children_and_own_aabbs(node_index, shapes);

            // Return parent node's index for upcoming refitting,
            // since this node just changed its `Aabb`.
            if node_index != 0 {
                Some(OptimizationIndex::Refit(parent_index))
            } else {
                None
            }
        } else {
            info!("    No useful rotation.");
            // Set the correct children `Aabb`s, which have been computed earlier.
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
                    // Otherwise, we still have to fix the parent's Aabbs
                    Some(OptimizationIndex::FixAabbs(parent_index))
                }
            } else {
                None
            }
        }
    }

    /// Sets `child_l_aabb` and child_r_aabb of a [`BvhNode::Node`] to match its children,
    /// right after updating the children themselves. Not recursive.
    fn fix_children_and_own_aabbs<Shape: BHShape<T, D>>(
        &mut self,
        node_index: usize,
        shapes: &[Shape],
    ) {
        let (child_l_index, child_r_index) = if let BvhNode::Node {
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

    /// Updates `child_l_aabb` and `child_r_aabb` of the `BvhNode::Node`
    /// with the index `node_index` from its children.
    fn fix_aabbs<Shape: BHShape<T, D>>(
        &mut self,
        node_index: usize,
        shapes: &[Shape],
    ) -> Option<OptimizationIndex> {
        match self.nodes[node_index] {
            BvhNode::Node {
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
                    Some(OptimizationIndex::FixAabbs(parent_index))
                } else {
                    None
                }
            }
            // Don't do anything if the node is a leaf.
            _ => None,
        }
    }

    /// Switch two nodes by rewiring the involved indices (not by moving them in the nodes slice).
    /// Also updates the [`Aabbs`]'s of the parents.
    fn rotate<Shape: BHShape<T, D>>(
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

    fn node_is_left_child(&self, node_index: usize) -> bool {
        // Get the index of the parent.
        let node_parent_index = self.nodes[node_index].parent();
        // Get the index of te left child of the parent.
        let child_l_index = self.nodes[node_parent_index].child_l();
        child_l_index == node_index
    }

    fn connect_nodes<Shape: BHShape<T, D>>(
        &mut self,
        child_index: usize,
        parent_index: usize,
        left_child: bool,
        shapes: &[Shape],
    ) {
        let child_aabb = self.nodes[child_index].get_node_aabb(shapes);
        info!("\tConnecting: {} < {}.", child_index, parent_index);
        // Set parent's child and `child_aabb`; and get its depth.
        let parent_depth = {
            match self.nodes[parent_index] {
                BvhNode::Node {
                    ref mut child_l_index,
                    ref mut child_r_index,
                    ref mut child_l_aabb,
                    ref mut child_r_aabb,
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
                }
                // Assuming that our `Bvh` is correct, the parent cannot be a leaf.
                _ => unreachable!(),
            }
        };

        // Set child's parent.
        *self.nodes[child_index].parent_mut() = parent_index;

    }

    /// Adds a shape with the given index to the `BVH`
    /// Significantly slower at building a `BVH` than the full build or rebuild option
    /// Useful for moving a small subset of nodes around in a large `BVH`
    pub fn add_node<Shape: BHShape<T, D>>(&mut self, shapes: &mut [Shape], new_shape_index: usize) where T: std::ops::Div<Output = T> {
        let mut i = 0;
        let new_shape = &shapes[new_shape_index];
        let shape_aabb = new_shape.aabb();
        let shape_sa = shape_aabb.surface_area();

        if self.nodes.is_empty() {
            self.nodes.push(BvhNode::Leaf {
                parent_index: 0,
                shape_index: new_shape_index,
            });
            shapes[new_shape_index].set_bh_node_index(0);
            return;
        }

        loop {
            match self.nodes[i] {
                BvhNode::Node {
                    child_l_aabb,
                    child_l_index,
                    child_r_aabb,
                    child_r_index,
                    parent_index,
                } => {
                    let left_expand = child_l_aabb.join(&shape_aabb);

                    let right_expand = child_r_aabb.join(&shape_aabb);

                    let send_left = child_r_aabb.surface_area() + left_expand.surface_area();
                    let send_right = child_l_aabb.surface_area() + right_expand.surface_area();
                    let merged_aabb = child_r_aabb.join(&child_l_aabb);
                    let merged = merged_aabb.surface_area() + shape_sa;

                    let merge_discount = 0.3;
                    
                    //dbg!(depth);

                    // compared SA of the options
                    let min_send = if send_left < send_right {
                        send_left
                    } else {
                         send_right
                    };
                    // merge is more expensive only do when it's significantly better

                    if merged < min_send * T::from_i8(3).unwrap() / T::from_i8(10).unwrap() {
                        //println!("Merging left and right trees");
                        // Merge left and right trees
                        let l_index = self.nodes.len();
                        let new_left = BvhNode::Leaf {
                            parent_index: i,
                            shape_index: new_shape_index,
                        };
                        shapes[new_shape_index].set_bh_node_index(l_index);
                        self.nodes.push(new_left);

                        let r_index = self.nodes.len();
                        let new_right = BvhNode::Node {
                            child_l_aabb,
                            child_l_index,
                            child_r_aabb,
                            child_r_index,
                            parent_index: i,
                        };
                        self.nodes.push(new_right);
                        *self.nodes[child_r_index].parent_mut() = r_index;
                        *self.nodes[child_l_index].parent_mut() = r_index;

                        self.nodes[i] = BvhNode::Node {
                            child_l_aabb: shape_aabb,
                            child_l_index: l_index,
                            child_r_aabb: merged_aabb,
                            child_r_index: r_index,
                            parent_index,
                        };
                        //self.fix_depth(l_index, depth + 1);
                        //self.fix_depth(r_index, depth + 1);
                        return;
                    } else if send_left < send_right {
                        // send new box down left side
                        //println!("Sending left");
                        if i == child_l_index {
                            panic!("broken loop");
                        }
                        let child_l_aabb = left_expand;
                        self.nodes[i] = BvhNode::Node {
                            child_l_aabb,
                            child_l_index,
                            child_r_aabb,
                            child_r_index,
                            parent_index,
                        };
                        i = child_l_index;
                    } else {
                        // send new box down right
                        //println!("Sending right");
                        if i == child_r_index {
                            panic!("broken loop");
                        }
                        let child_r_aabb = right_expand;
                        self.nodes[i] = BvhNode::Node {
                            child_l_aabb,
                            child_l_index,
                            child_r_aabb,
                            child_r_index,
                            parent_index,
                        };
                        i = child_r_index;
                    }
                }
                BvhNode::Leaf {
                    shape_index,
                    parent_index,
                } => {
                    //println!("Splitting leaf");
                    // Split leaf into 2 nodes and insert the new box
                    let l_index = self.nodes.len();
                    let new_left = BvhNode::Leaf {
                        parent_index: i,
                        shape_index: new_shape_index,
                    };
                    shapes[new_shape_index].set_bh_node_index(l_index);
                    self.nodes.push(new_left);

                    let child_r_aabb = shapes[shape_index].aabb();
                    let child_r_index = self.nodes.len();
                    let new_right = BvhNode::Leaf {
                        parent_index: i,
                        shape_index,
                    };
                    shapes[shape_index].set_bh_node_index(child_r_index);
                    self.nodes.push(new_right);

                    let new_node = BvhNode::Node {
                        child_l_aabb: shape_aabb,
                        child_l_index: l_index,
                        child_r_aabb,
                        child_r_index,
                        parent_index,
                    };
                    self.nodes[i] = new_node;
                    self.fix_aabbs_ascending(shapes, parent_index);
                    return;
                }
            }
        }
    }

    /// Removes a shape from the `BVH`
    /// if swap_shape is true, it swaps the shape you are removing with the last shape in the shape slice
    /// truncation of the data structure backing the shapes slice must be performed by the user
    pub fn remove_node<Shape: BHShape<T, D>>(
        &mut self,
        shapes: &mut [Shape],
        deleted_shape_index: usize,
        swap_shape: bool,
    ) {
        if self.nodes.is_empty() {
            return;
            //panic!("can't remove a node from a bvh with only one node");
        }
        let bad_shape = &shapes[deleted_shape_index];

        // to remove a node, delete it from the tree, remove the parent and replace it with the sibling
        // swap the node being removed to the end of the slice and adjust the index of the node that was removed
        // update the removed nodes index
        // swap the shape to the end and update the node to still point at the right shape
        let dead_node_index = bad_shape.bh_node_index();

        if self.nodes.len() == 1 {
            if dead_node_index == 0 {
                self.nodes.clear();
            }
        } else {
            //println!("delete_i={}", dead_node_index);

            let dead_node = self.nodes[dead_node_index];

            let parent_index = dead_node.parent();
            //println!("parent_i={}", parent_index);
            let gp_index = self.nodes[parent_index].parent();
            //println!("{}->{}->{}", gp_index, parent_index, dead_node_index);

            let sibling_index = if self.node_is_left_child(dead_node_index) {
                self.nodes[parent_index].child_r()
            } else {
                self.nodes[parent_index].child_l()
            };

            // TODO: fix potential issue leaving empty spot in self.nodes
            // the node swapped to sibling_index should probably be swapped to the end
            // of the vector and the vector truncated
            if parent_index == gp_index {
                // We are removing one of the children of the root node
                // The other child needs to become the root node
                // The old root node and the dead child then have to be moved

                if parent_index != 0 {
                    panic!(
                        "Circular node that wasn't root parent={} node={}",
                        parent_index, dead_node_index
                    );
                }

                match self.nodes[sibling_index] {
                    BvhNode::Node {
                        child_l_index,
                        child_r_index,
                        ..
                    } => {
                        self.connect_nodes(child_l_index, parent_index, true, shapes);
                        self.connect_nodes(child_r_index, parent_index, false, shapes);
                    }
                    _ => {
                        self.nodes[0] = self.nodes[sibling_index];
                        *self.nodes[0].parent_mut() = 0;
                        shapes[self.nodes[0].shape_index().unwrap()].set_bh_node_index(0);
                    }
                }

                self.swap_and_remove_index(shapes, sibling_index.max(dead_node_index));
                self.swap_and_remove_index(shapes, sibling_index.min(dead_node_index));
            } else {
                let parent_is_left = self.node_is_left_child(parent_index);

                self.connect_nodes(sibling_index, gp_index, parent_is_left, shapes);

                self.fix_aabbs_ascending(shapes, gp_index);
                self.swap_and_remove_index(shapes, dead_node_index.max(parent_index));
                self.swap_and_remove_index(shapes, parent_index.min(dead_node_index));
            }
        }

        if swap_shape {
            let end_shape = shapes.len() - 1;
            if deleted_shape_index < end_shape {
                shapes.swap(deleted_shape_index, end_shape);
                let node_index = shapes[deleted_shape_index].bh_node_index();
                if let Some(index) = self.nodes[node_index].shape_index_mut() {
                    *index = deleted_shape_index
                }
            }
        }
    }

    /// Fixes bvh
    pub fn optimize<'a, Shape: BHShape<T, D>>(
        &mut self,
        refit_shape_indices: impl IntoIterator<Item = &'a usize> + Copy,
        shapes: &mut [Shape],
    ) {
        for i in refit_shape_indices {
            self.remove_node(shapes, *i, false);
        }
        for i in refit_shape_indices {
            self.add_node(shapes, *i);
        }
    }

    fn fix_aabbs_ascending<Shape: BHShape<T, D>>(&mut self, shapes: &mut [Shape], node_index: usize) {
        let mut index_to_fix = node_index;
        while index_to_fix != 0 {
            let parent = self.nodes[index_to_fix].parent();
            match self.nodes[parent] {
                BvhNode::Node {
                    child_l_index,
                    child_r_index,
                    child_l_aabb,
                    child_r_aabb,
                    ..
                } => {
                    //println!("checking {} l={} r={}", parent, child_l_index, child_r_index);
                    let l_aabb = self.nodes[child_l_index].get_node_aabb(shapes);
                    let r_aabb = self.nodes[child_r_index].get_node_aabb(shapes);
                    //println!("child_l_aabb {}", l_aabb);
                    //println!("child_r_aabb {}", r_aabb);
                    let mut stop = true;
                    let epsilon = T::from_f32(0.00001).unwrap_or(T::zero());
                    if !l_aabb.relative_eq(&child_l_aabb, epsilon) {
                        stop = false;
                        //println!("setting {} l = {}", parent, l_aabb);
                        *self.nodes[parent].child_l_aabb_mut() = l_aabb;
                    }
                    if !r_aabb.relative_eq(&child_r_aabb, epsilon) {
                        stop = false;
                        //println!("setting {} r = {}", parent, r_aabb);
                        *self.nodes[parent].child_r_aabb_mut() = r_aabb;
                    }
                    if !stop {
                        index_to_fix = parent;
                        //dbg!(parent);
                    } else {
                        //dbg!(index_to_fix);
                        index_to_fix = 0;
                    }
                }
                _ => index_to_fix = 0,
            }
        }
    }
    
    fn swap_and_remove_index<Shape: BHShape<T, D>>(&mut self, shapes: &mut [Shape], node_index: usize) {
        let end = self.nodes.len() - 1;
        //println!("removing node {}", node_index);
        if node_index != end {
            self.nodes[node_index] = self.nodes[end];
            let parent_index = self.nodes[node_index].parent();

            if let BvhNode::Leaf { .. } = self.nodes[parent_index] {
                self.nodes.truncate(end);
                return;
            }
            let parent = self.nodes[parent_index];
            let moved_left = parent.child_l() == end;
            if !moved_left && parent.child_r() != end {
                self.nodes.truncate(end);
                return;
            }
            let ref_to_change = if moved_left {
                self.nodes[parent_index].child_l_mut()
            } else {
                self.nodes[parent_index].child_r_mut()
            };
            //println!("on {} changing {}=>{}", node_parent, ref_to_change, node_index);
            *ref_to_change = node_index;

            match self.nodes[node_index] {
                BvhNode::Leaf { shape_index, .. } => {
                    shapes[shape_index].set_bh_node_index(node_index);
                }
                BvhNode::Node {
                    child_l_index,
                    child_r_index,
                    ..
                } => {
                    *self.nodes[child_l_index].parent_mut() = node_index;
                    *self.nodes[child_r_index].parent_mut() = node_index;

                    //println!("{} {} {}", node_index, self.nodes[node_index].child_l_aabb(), self.nodes[node_index].child_r_aabb());
                    //let correct_depth
                    //self.fix_depth(child_l_index, )
                }
            }
        }
        self.nodes.truncate(end);
    }
}

#[cfg(test)]
mod tests {
    use crate::aabb::Bounded;
    use crate::bounding_hierarchy::BHShape;
    use crate::testbase::{
        build_some_bh, create_n_cubes, default_bounds, randomly_transform_scene, TBvh3, TBvhNode3,
        TPoint3, UnitBox,
    };
    use std::collections::HashSet;

    #[test]
    /// Tests if [`Bvh::optimize()`] does not modify a fresh [`Bvh`].
    fn test_optimizing_new_bvh() {
        let (mut shapes, mut bvh) = build_some_bh::<TBvh3>();
        let original_nodes = bvh.nodes.clone();

        // Query an update for all nodes.
        let refit_shape_indices: HashSet<usize> = (0..shapes.len()).collect();
        bvh.optimize(&refit_shape_indices, &mut shapes);

        // Assert that all nodes are the same as before the update.
        for (optimized, original) in bvh.nodes.iter().zip(original_nodes.iter()) {
            assert_eq!(optimized, original);
        }
    }


    #[test]
    /// Tests whether a Bvh is still consistent after a few optimization calls.
    fn test_consistent_after_optimize() {
        let (mut shapes, mut bvh) = build_some_bh::<TBvh3>();
        shapes[0].pos = TPoint3::new(10.0, 1.0, 2.0);
        shapes[1].pos = TPoint3::new(-10.0, -10.0, 10.0);
        shapes[2].pos = TPoint3::new(-10.0, 10.0, 10.0);
        shapes[3].pos = TPoint3::new(-10.0, 10.0, -10.0);
        shapes[4].pos = TPoint3::new(11.0, 1.0, 2.0);
        shapes[5].pos = TPoint3::new(11.0, 2.0, 2.0);

        let refit_shape_indices: Vec<_> = (0..6).collect();
        bvh.optimize(&refit_shape_indices, &mut shapes);
        bvh.assert_consistent(&shapes);
    }

    #[test]
    /// Test whether a simple update on a simple [`Bvh]` yields the expected optimization result.
    fn test_optimize_simple_update() {
        let mut shapes = vec![
            UnitBox::new(0, TPoint3::new(-50.0, 0.0, 0.0)),
            UnitBox::new(1, TPoint3::new(-40.0, 0.0, 0.0)),
            UnitBox::new(2, TPoint3::new(50.0, 0.0, 0.0)),
        ];

        let mut bvh = TBvh3::build(&mut shapes);
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
                    &TBvhNode3::Leaf {
                        parent_index: left_parent_index,
                        ..
                    },
                    &TBvhNode3::Leaf {
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
        shapes[1].pos = TPoint3::new(40.0, 0.0, 0.0);
        let refit_shape_indices: HashSet<usize> = (1..2).collect();
        bvh.optimize(&refit_shape_indices, &mut shapes);
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
                    &TBvhNode3::Leaf {
                        parent_index: right_parent_index,
                        ..
                    },
                    &TBvhNode3::Leaf {
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

    /// Creates a small [`Bvh`] with 4 shapes and 7 nodes.
    fn create_predictable_bvh() -> (Vec<UnitBox>, TBvh3) {
        let shapes = vec![
            UnitBox::new(0, TPoint3::new(0.0, 0.0, 0.0)),
            UnitBox::new(1, TPoint3::new(2.0, 0.0, 0.0)),
            UnitBox::new(2, TPoint3::new(4.0, 0.0, 0.0)),
            UnitBox::new(3, TPoint3::new(6.0, 0.0, 0.0)),
        ];

        let nodes = vec![
            // Root node.
            TBvhNode3::Node {
                parent_index: 0,
                child_l_aabb: shapes[0].aabb().join(&shapes[1].aabb()),
                child_l_index: 1,
                child_r_aabb: shapes[2].aabb().join(&shapes[3].aabb()),
                child_r_index: 2,
            },
            // Depth 1 nodes.
            TBvhNode3::Node {
                parent_index: 0,
                child_l_aabb: shapes[0].aabb(),
                child_l_index: 3,
                child_r_aabb: shapes[1].aabb(),
                child_r_index: 4,
            },
            TBvhNode3::Node {
                parent_index: 0,
                child_l_aabb: shapes[2].aabb(),
                child_l_index: 5,
                child_r_aabb: shapes[3].aabb(),
                child_r_index: 6,
            },
            // Depth 2 nodes (leaves).
            TBvhNode3::Leaf {
                parent_index: 1,
                shape_index: 0,
            },
            TBvhNode3::Leaf {
                parent_index: 1,
                shape_index: 1,
            },
            TBvhNode3::Leaf {
                parent_index: 2,
                shape_index: 2,
            },
            TBvhNode3::Leaf {
                parent_index: 2,
                shape_index: 3,
            },
        ];

        (shapes, TBvh3 { nodes })
    }

    #[test]
    fn test_connect_grandchildren() {
        let (shapes, mut bvh) = create_predictable_bvh();

        // Switch two nodes
        bvh.connect_nodes(3, 2, true, &shapes);
        bvh.connect_nodes(5, 1, true, &shapes);

        // Check if the resulting tree is as expected.
        let TBvh3 { nodes } = bvh;

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
            .relative_eq(&shapes[2].aabb(), f32::EPSILON));
        assert!(nodes[1]
            .child_r_aabb()
            .relative_eq(&shapes[1].aabb(), f32::EPSILON));
        assert!(nodes[2]
            .child_l_aabb()
            .relative_eq(&shapes[0].aabb(), f32::EPSILON));
        assert!(nodes[2]
            .child_r_aabb()
            .relative_eq(&shapes[3].aabb(), f32::EPSILON));
    }

    #[test]
    fn test_connect_child_grandchild() {
        let (shapes, mut bvh) = create_predictable_bvh();

        // Switch two nodes
        bvh.connect_nodes(1, 2, true, &shapes);
        bvh.connect_nodes(5, 0, true, &shapes);

        // Check if the resulting tree is as expected.
        let TBvh3 { nodes } = bvh;

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
            .relative_eq(&shapes[2].aabb(), f32::EPSILON));
        assert!(nodes[2]
            .child_r_aabb()
            .relative_eq(&shapes[3].aabb(), f32::EPSILON));
        assert!(nodes[1]
            .child_l_aabb()
            .relative_eq(&shapes[0].aabb(), f32::EPSILON));
        assert!(nodes[1]
            .child_r_aabb()
            .relative_eq(&shapes[1].aabb(), f32::EPSILON));
    }

    #[test]
    fn test_rotate_grandchildren() {
        let (shapes, mut bvh) = create_predictable_bvh();

        // Switch two nodes.
        bvh.rotate(3, 5, &shapes);

        // Check if the resulting tree is as expected.
        let TBvh3 { nodes } = bvh;

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
            .relative_eq(&shapes[2].aabb(), f32::EPSILON));
        assert!(nodes[1]
            .child_r_aabb()
            .relative_eq(&shapes[1].aabb(), f32::EPSILON));
        assert!(nodes[2]
            .child_l_aabb()
            .relative_eq(&shapes[0].aabb(), f32::EPSILON));
        assert!(nodes[2]
            .child_r_aabb()
            .relative_eq(&shapes[3].aabb(), f32::EPSILON));
    }

    #[test]
    fn test_rotate_child_grandchild() {
        let (shapes, mut bvh) = create_predictable_bvh();

        // Switch two nodes.
        bvh.rotate(1, 5, &shapes);

        // Check if the resulting tree is as expected.
        let TBvh3 { nodes } = bvh;

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
            .relative_eq(&shapes[2].aabb(), f32::EPSILON));
        assert!(nodes[2]
            .child_r_aabb()
            .relative_eq(&shapes[3].aabb(), f32::EPSILON));
        assert!(nodes[1]
            .child_l_aabb()
            .relative_eq(&shapes[0].aabb(), f32::EPSILON));
        assert!(nodes[1]
            .child_r_aabb()
            .relative_eq(&shapes[1].aabb(), f32::EPSILON));
    }

    #[test]
    fn test_try_rotate_child_grandchild() {
        let (mut shapes, mut bvh) = create_predictable_bvh();

        // Move the second shape.
        shapes[2].pos = TPoint3::new(-40.0, 0.0, 0.0);

        // Try to rotate node 2 because node 5 changed.
        bvh.try_rotate(2, &shapes);

        // Try to rotate node 0 because rotating node 2 should not have yielded a result.
        bvh.try_rotate(0, &shapes);

        // Check if the resulting tree is as expected.
        let TBvh3 { nodes } = bvh;

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
            .relative_eq(&shapes[2].aabb(), f32::EPSILON));
        let right_subtree_aabb = &shapes[0]
            .aabb()
            .join(&shapes[1].aabb())
            .join(&shapes[3].aabb());
        assert!(nodes[0]
            .child_r_aabb()
            .relative_eq(right_subtree_aabb, f32::EPSILON));

        assert!(nodes[2]
            .child_r_aabb()
            .relative_eq(&shapes[3].aabb(), f32::EPSILON));
        assert!(nodes[1]
            .child_l_aabb()
            .relative_eq(&shapes[0].aabb(), f32::EPSILON));
        assert!(nodes[1]
            .child_r_aabb()
            .relative_eq(&shapes[1].aabb(), f32::EPSILON));
    }

    #[test]
    /// Test optimizing [`Bvh`] after randomizing 50% of the shapes.
    fn test_optimize_bvh_12k_75p() {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(1_000, &bounds);

        let mut bvh = TBvh3::build(&mut triangles);

        // The initial Bvh should be consistent.
        bvh.assert_consistent(&triangles);
        bvh.assert_tight();

        // After moving triangles, the Bvh should be inconsistent, because the shape `Aabb`s do not
        // match the tree entries.
        let mut seed = 0;

        let updated = randomly_transform_scene(&mut triangles, 9_000, &bounds, None, &mut seed);
        assert!(!bvh.is_consistent(&triangles), "Bvh is consistent.");

        // After fixing the `Aabb` consistency should be restored.
        bvh.optimize(&updated, &mut triangles);
        bvh.assert_consistent(&triangles);
        bvh.assert_tight();
    }
}

#[cfg(all(feature = "bench", test))]
mod bench {
    use crate::testbase::{
        create_n_cubes, default_bounds, intersect_bh, load_sponza_scene, randomly_transform_scene,
        TAabb3, TBvh3, Triangle,
    };

    #[bench]
    /// Benchmark randomizing 50% of the shapes in a [`Bvh`].
    fn bench_randomize_120k_50p(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        let mut seed = 0;

        b.iter(|| {
            randomly_transform_scene(&mut triangles, 60_000, &bounds, None, &mut seed);
        });
    }

    /// Benchmark optimizing a [`Bvh`] with 120,000 [`Triangle`]'ss, where `percent`
    /// [`Triangle`]'s have been randomly moved.
    fn optimize_bvh_120k(percent: f32, b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        let mut bvh = TBvh3::build(&mut triangles);
        let num_move = (triangles.len() as f32 * percent) as usize;
        let mut seed = 0;

        b.iter(|| {
            let updated =
                randomly_transform_scene(&mut triangles, num_move, &bounds, Some(10.0), &mut seed);
            bvh.optimize(&updated, &mut triangles);
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

    /// Move `percent` [`Triangle`]`s in the scene given by `triangles` and optimize the
    /// [`Bvh`]. Iterate this procedure `iterations` times. Afterwards benchmark the performance
    /// of intersecting this scene/[`Bvh`].
    fn intersect_scene_after_optimize(
        triangles: &mut Vec<Triangle>,
        bounds: &TAabb3,
        percent: f32,
        max_offset: Option<f32>,
        iterations: usize,
        b: &mut ::test::Bencher,
    ) {
        let mut bvh = TBvh3::build(triangles);
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

    /// Move `percent` [`Triangle`]'s in the scene given by `triangles` `iterations` times.
    /// Afterwards optimize the `Bvh` and benchmark the performance of intersecting this
    /// scene/[`Bvh`]. Used to compare optimizing with rebuilding. For reference see
    /// `intersect_scene_after_optimize`.
    fn intersect_scene_with_rebuild(
        triangles: &mut Vec<Triangle>,
        bounds: &TAabb3,
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

        let bvh = TBvh3::build(triangles);
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

    /// Benchmark intersecting a [`Bvh`] for Sponza after randomly moving one [`Triangle`] and
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

    /// Benchmark intersecting a [`Bvh`] for Sponza after rebuilding. Used to compare optimizing
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
