//! This module defines the optimization function for the [`Bvh`].
//! By passing the indices of shapes that have changed, the function determines possible
//! tree rotations and optimizes the Bvh using a SAH.
//! Based on [`https://www.sci.utah.edu/~thiago/papers/rotations.pdf`]
//!
//! [`Bvh`]: struct.Bvh.html
//!

use crate::bounding_hierarchy::{BHShape, BHValue};
use crate::bvh::*;

// TODO Consider: Instead of getting the scene's shapes passed, let leaf nodes store an `Aabb`
// that is updated from the outside, perhaps by passing not only the indices of the changed
// shapes, but also their new `Aabb`'s into update_shapes().
// TODO Consider: Stop updating `Aabb`'s upwards the tree once an `Aabb` didn't get changed.

impl<T: BHValue, const D: usize> Bvh<T, D> {
    fn node_is_left_child(&self, node_index: usize) -> bool {
        // Get the index of the parent.
        let node_parent_index = self.nodes[node_index].parent();
        // Get the index of te left child of the parent.
        let child_l_index = self.nodes[node_parent_index].child_l();
        child_l_index == node_index
    }

    fn node_is_right_child(&self, node_index: usize) -> bool {
        // Get the index of the parent.
        let node_parent_index = self.nodes[node_index].parent();
        // Get the index of te right child of the parent.
        let child_r_index = self.nodes[node_parent_index].child_r();
        child_r_index == node_index
    }

    fn connect_nodes<Shape: BHShape<T, D>>(
        &mut self,
        child_index: usize,
        parent_index: usize,
        left_child: bool,
        shapes: &[Shape],
    ) {
        let child_aabb = self.nodes[child_index].get_node_aabb(shapes);
        // Set parent's child aabb and index.
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
            }
            // Assuming that our `Bvh` is correct, the parent cannot be a leaf.
            _ => unreachable!(),
        }

        // Set child's parent.
        *self.nodes[child_index].parent_mut() = parent_index;
    }

    /// Adds a shape with the given index to the `BVH`
    /// Significantly slower at building a `BVH` than the full build or rebuild option
    /// Useful for moving a small subset of nodes around in a large `BVH`
    pub fn add_shape<Shape: BHShape<T, D>>(&mut self, shapes: &mut [Shape], new_shape_index: usize)
    where
        T: core::ops::Div<Output = T>,
    {
        let mut node_index = 0;
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
            match self.nodes[node_index] {
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

                    // compared SA of the options
                    let min_send = if send_left < send_right {
                        send_left
                    } else {
                        send_right
                    };
                    // merge is more expensive only do when it's significantly better

                    if merged < min_send * T::from_i8(3).unwrap() / T::from_i8(10).unwrap() {
                        // Merge left and right trees
                        let l_index = self.nodes.len();
                        let new_left = BvhNode::Leaf {
                            parent_index: node_index,
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
                            parent_index: node_index,
                        };
                        self.nodes.push(new_right);
                        *self.nodes[child_r_index].parent_mut() = r_index;
                        *self.nodes[child_l_index].parent_mut() = r_index;

                        self.nodes[node_index] = BvhNode::Node {
                            child_l_aabb: shape_aabb,
                            child_l_index: l_index,
                            child_r_aabb: merged_aabb,
                            child_r_index: r_index,
                            parent_index,
                        };
                        return;
                    } else if send_left < send_right {
                        // Send new box down left side
                        if node_index == child_l_index {
                            panic!("broken loop");
                        }
                        let child_l_aabb = left_expand;
                        self.nodes[node_index] = BvhNode::Node {
                            child_l_aabb,
                            child_l_index,
                            child_r_aabb,
                            child_r_index,
                            parent_index,
                        };
                        node_index = child_l_index;
                    } else {
                        // Send new box down right
                        if node_index == child_r_index {
                            panic!("broken loop");
                        }
                        let child_r_aabb = right_expand;
                        self.nodes[node_index] = BvhNode::Node {
                            child_l_aabb,
                            child_l_index,
                            child_r_aabb,
                            child_r_index,
                            parent_index,
                        };
                        node_index = child_r_index;
                    }
                }
                BvhNode::Leaf {
                    shape_index,
                    parent_index,
                } => {
                    // Split leaf into 2 nodes and insert the new box
                    let l_index = self.nodes.len();
                    let new_left = BvhNode::Leaf {
                        parent_index: node_index,
                        shape_index: new_shape_index,
                    };
                    shapes[new_shape_index].set_bh_node_index(l_index);
                    self.nodes.push(new_left);

                    let child_r_aabb = shapes[shape_index].aabb();
                    let child_r_index = self.nodes.len();
                    let new_right = BvhNode::Leaf {
                        parent_index: node_index,
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
                    self.nodes[node_index] = new_node;
                    self.fix_aabbs_ascending(shapes, parent_index);
                    return;
                }
            }
        }
    }

    /// Removes a shape from the `BVH`
    /// if swap_shape is true, it swaps the shape you are removing with the last shape in the shape slice
    /// truncation of the data structure backing the shapes slice must be performed by the user
    pub fn remove_shape<Shape: BHShape<T, D>>(
        &mut self,
        shapes: &mut [Shape],
        deleted_shape_index: usize,
        swap_shape: bool,
    ) {
        if self.nodes.is_empty() {
            panic!("can't remove a node from a bvh with only one node");
        }
        let bad_shape = &shapes[deleted_shape_index];

        // to remove a node, delete it from the tree, remove the parent and replace it with the sibling
        // swap the node being removed to the end of the slice and adjust the index of the node that was removed
        // update the removed nodes index
        // swap the shape to the end and update the node to still point at the right shape
        let dead_node_index = bad_shape.bh_node_index();

        if self.nodes.len() == 1 {
            assert_eq!(dead_node_index, 0);
            assert!(self.nodes[0].is_leaf());
            self.nodes.clear();
        } else {
            let dead_node = self.nodes[dead_node_index];
            assert!(dead_node.is_leaf());

            let parent_index = dead_node.parent();
            let gp_index = self.nodes[parent_index].parent();

            let sibling_index = if self.node_is_left_child(dead_node_index) {
                self.nodes[parent_index].child_r()
            } else {
                assert!(self.node_is_right_child(dead_node_index));
                self.nodes[parent_index].child_l()
            };

            // TODO: fix potential issue leaving empty spot in self.nodes
            // the node swapped to sibling_index should probably be swapped to the end
            // of the vector and the vector truncated
            if parent_index == gp_index {
                // We are removing one of the children of the root node
                // The other child needs to become the root node
                // The old root node and the dead child then have to be removed
                assert_eq!(
                    parent_index, 0,
                    "Circular node that wasn't root parent={parent_index} node={dead_node_index}"
                );

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

                // Remove in decreasing order of index.
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
                *self.nodes[node_index]
                    .shape_index_mut()
                    .expect("bh_node_index() must be a leaf node") = deleted_shape_index;
            }
        }
    }

    /// Fixes bvh
    pub fn update_shapes<'a, Shape: BHShape<T, D>>(
        &mut self,
        changed_shape_indices: impl IntoIterator<Item = &'a usize> + Copy,
        shapes: &mut [Shape],
    ) {
        for i in changed_shape_indices {
            self.remove_shape(shapes, *i, false);
        }
        for i in changed_shape_indices {
            self.add_shape(shapes, *i);
        }
    }

    fn fix_aabbs_ascending<Shape: BHShape<T, D>>(&mut self, shapes: &[Shape], node_index: usize) {
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
                    let l_aabb = self.nodes[child_l_index].get_node_aabb(shapes);
                    let r_aabb = self.nodes[child_r_index].get_node_aabb(shapes);
                    let mut stop = true;
                    // Avoid `relative_eq`, because rounding errors can accumulate and
                    // eventually the BVH won't necessarily be tight.
                    if l_aabb != child_l_aabb {
                        stop = false;
                        *self.nodes[parent].child_l_aabb_mut() = l_aabb;
                    }
                    if r_aabb != child_r_aabb {
                        stop = false;
                        *self.nodes[parent].child_r_aabb_mut() = r_aabb;
                    }
                    if !stop {
                        index_to_fix = parent;
                    } else {
                        index_to_fix = 0;
                    }
                }
                _ => index_to_fix = 0,
            }
        }
    }

    fn swap_and_remove_index<Shape: BHShape<T, D>>(
        &mut self,
        shapes: &mut [Shape],
        node_index: usize,
    ) {
        let end = self.nodes.len() - 1;
        if node_index != end {
            self.nodes[node_index] = self.nodes[end];
            let parent_index = self.nodes[node_index].parent();

            let parent = self.nodes[parent_index];
            assert!(!parent.is_leaf());
            let moved_left = parent.child_l() == end;
            let ref_to_change = if moved_left {
                self.nodes[parent_index].child_l_mut()
            } else {
                assert_eq!(parent.child_r(), end);
                self.nodes[parent_index].child_r_mut()
            };
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
                }
            }
        }
        self.nodes.truncate(end);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aabb::{Aabb, Bounded};
    use alloc::vec;
    use alloc::vec::Vec;

    type TBvh3 = Bvh<f32, 3>;
    type TBvhNode3 = BvhNode<f32, 3>;
    type TPoint3 = nalgebra::Point3<f32>;

    // Minimal local shape; keeps this white-box test independent of the shared
    // `bvh-testutil` helper crate (which would fork `bvh` in the test build).
    #[derive(PartialEq, Debug)]
    struct UnitBox {
        pos: TPoint3,
        node_index: usize,
    }

    impl UnitBox {
        fn new(_id: i32, pos: TPoint3) -> UnitBox {
            UnitBox { pos, node_index: 0 }
        }
    }

    impl Bounded<f32, 3> for UnitBox {
        fn aabb(&self) -> Aabb<f32, 3> {
            let min = self.pos + nalgebra::Vector3::new(-0.5, -0.5, -0.5);
            let max = self.pos + nalgebra::Vector3::new(0.5, 0.5, 0.5);
            Aabb::with_bounds(min, max)
        }
    }

    impl BHShape<f32, 3> for UnitBox {
        fn set_bh_node_index(&mut self, index: usize) {
            self.node_index = index;
        }
        fn bh_node_index(&self) -> usize {
            self.node_index
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

        assert!(
            nodes[1]
                .child_l_aabb()
                .relative_eq(&shapes[2].aabb(), f32::EPSILON)
        );
        assert!(
            nodes[1]
                .child_r_aabb()
                .relative_eq(&shapes[1].aabb(), f32::EPSILON)
        );
        assert!(
            nodes[2]
                .child_l_aabb()
                .relative_eq(&shapes[0].aabb(), f32::EPSILON)
        );
        assert!(
            nodes[2]
                .child_r_aabb()
                .relative_eq(&shapes[3].aabb(), f32::EPSILON)
        );
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

        assert!(
            nodes[0]
                .child_l_aabb()
                .relative_eq(&shapes[2].aabb(), f32::EPSILON)
        );
        assert!(
            nodes[2]
                .child_r_aabb()
                .relative_eq(&shapes[3].aabb(), f32::EPSILON)
        );
        assert!(
            nodes[1]
                .child_l_aabb()
                .relative_eq(&shapes[0].aabb(), f32::EPSILON)
        );
        assert!(
            nodes[1]
                .child_r_aabb()
                .relative_eq(&shapes[1].aabb(), f32::EPSILON)
        );
    }
}
