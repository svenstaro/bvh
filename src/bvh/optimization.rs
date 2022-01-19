//! This module defines the optimization function for the [`BVH`].
//! By passing the indices of shapes that have changed, the function determines possible
//! tree rotations and optimizes the BVH using a SAH.
//! Based on http://www.cs.utah.edu/~thiago/papers/rotations.pdf
//!
//! [`BVH`]: struct.BVH.html
//!

use crate::bounding_hierarchy::BHShape;

use crate::{bvh::*, EPSILON};

use log::info;

impl BVH {
    /// Optimizes the `BVH` by batch-reorganizing updated nodes.
    /// Based on https://github.com/jeske/SimpleScene/blob/master/SimpleScene/Util/ssBVH/ssBVH.cs
    ///
    /// Needs all the scene's shapes, plus the indices of the shapes that were updated.
    ///
    #[cfg(not(feature = "serde_impls"))]
    pub fn optimize<'a, Shape: BHShape>(
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

    /// Optimizes the `BVH` by batch-reorganizing updated nodes.
    /// Based on https://github.com/jeske/SimpleScene/blob/master/SimpleScene/Util/ssBVH/ssBVH.cs
    ///
    /// Needs all the scene's shapes, plus the indices of the shapes that were updated.
    ///
    #[cfg(feature = "serde_impls")]
    pub fn optimize<'a, Shape: BHShape + Clone + serde::Serialize>(
        &mut self,
        refit_shape_indices: impl IntoIterator<Item = &'a usize> + Copy,
        shapes: &mut [Shape],
    ) {
        //let mut prev_aabb_one = self.nodes[1].child_l_aabb();
        // let prev = self.clone();
        // let prev_shapes = shapes.to_owned();
        // let refit_shapes: Vec<_> = refit_shape_indices.into_iter().map(|&x| x).collect();
        for i in refit_shape_indices {
            //println!("removing shape {}", i);
            self.remove_node(shapes, *i, false);
            // if self.nodes[1].child_l_aabb() != prev_aabb_one {
            //     dbg!(&self.nodes[1].child_l_aabb());
            //     prev_aabb_one = self.nodes[1].child_l_aabb();
            //     dbg!(i);
            // }
            //self.assert_tight(shapes);
        }

        // if !self.is_consistent(shapes) {
        //     let bvh = serde_json::to_string_pretty(&prev).expect("bvh to serialize");
        //     let shapes_str = serde_json::to_string_pretty(&prev_shapes).expect("shapes to serialize");
        //     let refits_str = serde_json::to_string_pretty(&refit_shapes).expect("shapes to serialize");
        //     dbg!(bvh.len());
        //     std::fs::write("badbvh.json", &bvh).expect("unable to write to file");
        //     std::fs::write("badshapes.json", &shapes_str).expect("unable to write to file");
        //     std::fs::write("badrefits.json", &refits_str).expect("unable to write to file");
        // }

        self.assert_consistent(shapes);
        //println!("--------");
        //self.pretty_print();
        //println!("--------");
        for i in refit_shape_indices {
            //self.assert_tight(shapes);
            //println!("--------");
            //self.pretty_print();
            //println!("--------");
            //self.assert_consistent(shapes);
            // let prev = self.clone();
            // let prev_shapes = shapes.to_owned();
            self.add_node(shapes, *i);
            // if !self.is_consistent(shapes) {
            //     let bvh = serde_json::to_string_pretty(&prev).expect("bvh to serialize");
            //     let shapes_str = serde_json::to_string_pretty(&prev_shapes).expect("shapes to serialize");
            //     dbg!(bvh.len());
            //     std::fs::write("badbvh.json", &bvh).expect("unable to write to file");
            //     std::fs::write("badshapes.json", &shapes_str).expect("unable to write to file");

            //     dbg!(i);
            //     dbg!(&shapes[*i].aabb());
            //     dbg!(&shapes[*i].bh_node_index());
            // self.assert_consistent(shapes);
            // }
        }
    }

    /// Adds a shape with the given index to the `BVH`
    /// Significantly slower at building a `BVH` than the full build or rebuild option
    /// Useful for moving a small subset of nodes around in a large `BVH`
    pub fn add_node<T: BHShape>(&mut self, shapes: &mut [T], new_shape_index: usize) {
        let mut i = 0;
        let new_shape = &shapes[new_shape_index];
        let shape_aabb = new_shape.aabb();
        let shape_sa = shape_aabb.surface_area();

        if self.nodes.len() == 0 {
            self.nodes.push(BVHNode::Leaf {
                parent_index: 0,
                shape_index: new_shape_index,
            });
            shapes[new_shape_index].set_bh_node_index(0);
            return;
        }

        loop {
            match self.nodes[i] {
                BVHNode::Node {
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

                    // merge is more expensive only do when it's significantly better
                    let merge_discount = 0.3;
                    //dbg!(depth);

                    // compared SA of the options
                    if merged < send_left.min(send_right) * merge_discount {
                        //println!("Merging left and right trees");
                        // Merge left and right trees
                        let l_index = self.nodes.len();
                        let new_left = BVHNode::Leaf {
                            parent_index: i,
                            shape_index: new_shape_index,
                        };
                        shapes[new_shape_index].set_bh_node_index(l_index);
                        self.nodes.push(new_left);

                        let r_index = self.nodes.len();
                        let new_right = BVHNode::Node {
                            child_l_aabb: child_l_aabb,
                            child_l_index,
                            child_r_aabb: child_r_aabb.clone(),
                            child_r_index,
                            parent_index: i,
                        };
                        self.nodes.push(new_right);
                        *self.nodes[child_r_index].parent_mut() = r_index;
                        *self.nodes[child_l_index].parent_mut() = r_index;

                        self.nodes[i] = BVHNode::Node {
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
                        self.nodes[i] = BVHNode::Node {
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
                        self.nodes[i] = BVHNode::Node {
                            child_l_aabb,
                            child_l_index,
                            child_r_aabb,
                            child_r_index,
                            parent_index,
                        };
                        i = child_r_index;
                    }
                }
                BVHNode::Leaf {
                    shape_index,
                    parent_index,
                } => {
                    //println!("Splitting leaf");
                    // Split leaf into 2 nodes and insert the new box
                    let l_index = self.nodes.len();
                    let new_left = BVHNode::Leaf {
                        parent_index: i,
                        shape_index: new_shape_index,
                    };
                    shapes[new_shape_index].set_bh_node_index(l_index);
                    self.nodes.push(new_left);

                    let child_r_aabb = shapes[shape_index].aabb();
                    let child_r_index = self.nodes.len();
                    let new_right = BVHNode::Leaf {
                        parent_index: i,
                        shape_index: shape_index,
                    };
                    shapes[shape_index].set_bh_node_index(child_r_index);
                    self.nodes.push(new_right);

                    let new_node = BVHNode::Node {
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
    pub fn remove_node<T: BHShape>(
        &mut self,
        shapes: &mut [T],
        deleted_shape_index: usize,
        swap_shape: bool,
    ) {
        if self.nodes.len() == 0 {
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
            //dbg!((gp_index, parent_index, dead_node_index, sibling_index));

            // TODO: fix potential issue leaving empty spot in self.nodes
            // the node swapped to sibling_index should probably be swapped to the end
            // of the vector and the vector truncated
            if parent_index == gp_index {
                // We are removing one of the children of the root node
                // The other child needs to become the root node
                // The old root node and the dead child then have to be moved

                // println!("gp == parent {}", parent_index);
                if parent_index != 0 {
                    panic!(
                        "Circular node that wasn't root parent={} node={}",
                        parent_index, dead_node_index
                    );
                }

                match self.nodes[sibling_index] {
                    BVHNode::Node {
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
                //println!("nodes_len {}, sib_index {}", self.nodes.len(), sibling_index);
                //println!("nodes_len {}", self.nodes.len());
            } else {
                let parent_is_left = self.node_is_left_child(parent_index);

                self.connect_nodes(sibling_index, gp_index, parent_is_left, shapes);

                self.fix_aabbs_ascending(shapes, gp_index);
                //let new_depth = self.nodes[sibling_index].depth() - 1;
                //*self.nodes[sibling_index].depth_mut() = new_depth;
                // remove node and parent

                //println!("---");
                //self.pretty_print();
                //println!("---");
                self.swap_and_remove_index(shapes, dead_node_index.max(parent_index));

                //println!("---");
                //self.pretty_print();
                //println!("---");
                self.swap_and_remove_index(shapes, parent_index.min(dead_node_index));

                //println!("---");
                //self.pretty_print();
                //println!("---");
            }
        }

        if swap_shape {
            let end_shape = shapes.len() - 1;
            if deleted_shape_index < end_shape {
                shapes.swap(deleted_shape_index, end_shape);
                let node_index = shapes[deleted_shape_index].bh_node_index();
                match self.nodes[node_index].shape_index_mut() {
                    Some(index) => *index = deleted_shape_index,
                    _ => {}
                }
            }
        }
    }

    fn fix_aabbs_ascending<T: BHShape>(&mut self, shapes: &mut [T], node_index: usize) {
        let mut index_to_fix = node_index;
        while index_to_fix != 0 {
            let parent = self.nodes[index_to_fix].parent();
            match self.nodes[parent] {
                BVHNode::Node {
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
                    if !l_aabb.relative_eq(&child_l_aabb, EPSILON) {
                        stop = false;
                        //println!("setting {} l = {}", parent, l_aabb);
                        *self.nodes[parent].child_l_aabb_mut() = l_aabb;
                    }
                    if !r_aabb.relative_eq(&child_r_aabb, EPSILON) {
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

    fn swap_and_remove_index<T: BHShape>(&mut self, shapes: &mut [T], node_index: usize) {
        let end = self.nodes.len() - 1;
        //println!("removing node {}", node_index);
        if node_index != end {
            self.nodes[node_index] = self.nodes[end];
            let parent_index = self.nodes[node_index].parent();
            match self.nodes[parent_index] {
                BVHNode::Leaf { .. } => {
                    // println!(
                    //     "truncating early node_parent={} parent_index={} shape_index={}",
                    //     node_parent, parent_index, shape_index
                    // );
                    self.nodes.truncate(end);
                    return;
                }
                _ => {}
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
                BVHNode::Leaf { shape_index, .. } => {
                    shapes[shape_index].set_bh_node_index(node_index);
                }
                BVHNode::Node {
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

    fn node_is_left_child(&self, node_index: usize) -> bool {
        // Get the index of the parent.
        let node_parent_index = self.nodes[node_index].parent();
        // Get the index of the left child of the parent.
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
        if child_index == parent_index {
            return;
        }
        let child_aabb = self.nodes[child_index].get_node_aabb(shapes);
        //info!("\tConnecting: {} < {}.", child_index, parent_index);
        // Set parent's child and child_aabb; and get its depth.
        let _ = {
            match self.nodes[parent_index] {
                BVHNode::Node {
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
                // Assuming that our BVH is correct, the parent cannot be a leaf.
                _ => unreachable!(),
            }
        };

        // Set child's parent.
        *self.nodes[child_index].parent_mut() = parent_index;
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

        let refit_shape_indices: Vec<usize> = (0..6).collect();
        bvh.optimize(&refit_shape_indices, &mut shapes);
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
        let refit_shape_indices: Vec<usize> = (1..2).collect();
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
                child_l_aabb: shapes[0].aabb().join(&shapes[1].aabb()),
                child_l_index: 1,
                child_r_aabb: shapes[2].aabb().join(&shapes[3].aabb()),
                child_r_index: 2,
            },
            // Depth 1 nodes.
            BVHNode::Node {
                parent_index: 0,
                child_l_aabb: shapes[0].aabb(),
                child_l_index: 3,
                child_r_aabb: shapes[1].aabb(),
                child_r_index: 4,
            },
            BVHNode::Node {
                parent_index: 0,
                child_l_aabb: shapes[2].aabb(),
                child_l_index: 5,
                child_r_aabb: shapes[3].aabb(),
                child_r_index: 6,
            },
            // Depth 2 nodes (leaves).
            BVHNode::Leaf {
                parent_index: 1,
                shape_index: 0,
            },
            BVHNode::Leaf {
                parent_index: 1,
                shape_index: 1,
            },
            BVHNode::Leaf {
                parent_index: 2,
                shape_index: 2,
            },
            BVHNode::Leaf {
                parent_index: 2,
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
    /// Test optimizing `BVH` after randomizing 50% of the shapes.
    fn test_optimize_bvh_12k_75p() {
        let c = 1000;
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(c, &bounds);

        let mut bvh = BVH::build(&mut triangles);

        // The initial BVH should be consistent.
        bvh.assert_consistent(&triangles);
        bvh.assert_tight(&triangles);

        // After moving triangles, the BVH should be inconsistent, because the shape `AABB`s do not
        // match the tree entries.
        let mut seed = 0;

        let updated = randomly_transform_scene(&mut triangles, c * 9, &bounds, None, &mut seed);
        assert!(!bvh.is_consistent(&triangles), "BVH is consistent.");

        // After fixing the `AABB` consistency should be restored.
        println!("optimize");
        bvh.optimize(&updated, &mut triangles);
        bvh.assert_consistent(&triangles);
        bvh.assert_tight(&triangles);
    }

    #[test]
    /// Test optimizing `BVH` after randomizing 100% of the shapes.
    fn test_optimize_bvh_12k_100p() {
        let c = 1000;
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(c, &bounds);

        let mut bvh = BVH::build(&mut triangles);

        // The initial BVH should be consistent.
        bvh.assert_consistent(&triangles);
        bvh.assert_tight(&triangles);

        // After moving triangles, the BVH should be inconsistent, because the shape `AABB`s do not
        // match the tree entries.
        let mut seed = 0;

        let updated = randomly_transform_scene(&mut triangles, c * 12, &bounds, None, &mut seed);
        assert!(!bvh.is_consistent(&triangles), "BVH is consistent.");

        // After fixing the `AABB` consistency should be restored.
        println!("optimize");
        bvh.optimize(&updated, &mut triangles);
        bvh.assert_consistent(&triangles);
        bvh.assert_tight(&triangles);
    }

    #[test]
    /// Test optimizing `BVH` after randomizing 100% of the shapes.
    fn test_shapes_reachable_optimize_bvh_1200_100p() {
        let c = 1000;
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(c, &bounds);

        let mut bvh = BVH::build(&mut triangles);

        // The initial BVH should be consistent.
        bvh.assert_consistent(&triangles);
        bvh.assert_tight(&triangles);

        // After moving triangles, the BVH should be inconsistent, because the shape `AABB`s do not
        // match the tree entries.
        let mut seed = 0;

        let updated = randomly_transform_scene(&mut triangles, c * 12, &bounds, None, &mut seed);
        assert!(!bvh.is_consistent(&triangles), "BVH is consistent.");

        // After fixing the `AABB` consistency should be restored.
        println!("optimize");
        bvh.optimize(&updated, &mut triangles);
        bvh.assert_consistent(&triangles);
        bvh.assert_tight(&triangles);
        bvh.assert_reachable(&triangles);
    }

    #[test]
    #[cfg(feature = "serde_impls")]
    fn test_bad_bvh() {
        let bvh_str = std::fs::read_to_string("bvh_2.json").expect("unable to read file");
        let refit_str = std::fs::read_to_string("refitshapes_2.json").expect("unable to read file");
        let shapes_str = std::fs::read_to_string("shapes_2.json").expect("unable to read file");

        let mut bvh: BVH = serde_json::from_str(&bvh_str).expect("to parse");
        dbg!(&bvh.nodes[1]);
        let mut shapes: Vec<Triangle> = serde_json::from_str(&shapes_str).expect("to parse");
        let refit_shapes: Vec<usize> = serde_json::from_str(&refit_str).expect("to parse");
        for (i, shape) in shapes.iter().enumerate() {
            let bh_index = shape.bh_node_index();
            let node = bvh.nodes[bh_index];
            let parent = bvh.nodes[node.parent()];
            let bh_aabb = if bvh.node_is_left_child(bh_index) {
                parent.child_l_aabb()
            } else {
                parent.child_r_aabb()
            };
            if refit_shapes.contains(&i) {
                assert!(!bh_aabb.relative_eq(&shape.aabb(), EPSILON))
            } else {
                assert!(bh_aabb.relative_eq(&shape.aabb(), EPSILON))
            }
        }
        bvh.optimize(&refit_shapes, &mut shapes);
    }

    #[test]
    fn test_optimize_bvh_12_75p() {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10, &bounds);
        println!("triangles={}", triangles.len());

        let mut bvh = BVH::build(&mut triangles);

        // The initial BVH should be consistent.
        bvh.assert_consistent(&triangles);
        bvh.assert_tight(&triangles);

        // After moving triangles, the BVH should be inconsistent, because the shape `AABB`s do not
        // match the tree entries.
        let mut seed = 0;

        let updated = randomly_transform_scene(&mut triangles, 90, &bounds, None, &mut seed);
        assert!(!bvh.is_consistent(&triangles), "BVH is consistent.");
        println!("triangles={}", triangles.len());
        //bvh.pretty_print();

        // After fixing the `AABB` consistency should be restored.
        bvh.optimize(&updated, &mut triangles);
        //bvh.pretty_print();
        bvh.assert_consistent(&triangles);
        bvh.assert_tight(&triangles);
    }

    #[test]
    fn test_optimizing_nodes() {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(1, &bounds);
        println!("triangles={}", triangles.len());

        let mut bvh = BVH::build(&mut triangles);

        // The initial BVH should be consistent.
        bvh.assert_consistent(&triangles);
        bvh.assert_tight(&triangles);

        // After moving triangles, the BVH should be inconsistent, because the shape `AABB`s do not
        // match the tree entries.
        let mut seed = 0;

        for _ in 0..1000 {
            let updated = randomly_transform_scene(&mut triangles, 1, &bounds, None, &mut seed);
            assert!(!bvh.is_consistent(&triangles), "BVH is consistent.");
            //bvh.pretty_print();

            // After fixing the `AABB` consistency should be restored.
            bvh.optimize(&updated, &mut triangles);
            //bvh.pretty_print();
            bvh.assert_consistent(&triangles);
            bvh.assert_tight(&triangles);
        }
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
    use crate::Real;

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
    fn optimize_bvh_120k(percent: Real, b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        let mut bvh = BVH::build(&mut triangles);
        let num_move = (triangles.len() as Real * percent) as usize;
        let mut seed = 0;

        b.iter(|| {
            let updated =
                randomly_transform_scene(&mut triangles, num_move, &bounds, Some(10.0), &mut seed);
            let updated: Vec<usize> = updated.into_iter().collect();
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

    #[bench]
    fn bench_optimize_bvh_120k_100p(b: &mut ::test::Bencher) {
        optimize_bvh_120k(1.0, b);
    }

    /// Move `percent` `Triangle`s in the scene given by `triangles` and optimize the
    /// `BVH`. Iterate this procedure `iterations` times. Afterwards benchmark the performance
    /// of intersecting this scene/`BVH`.
    fn intersect_scene_after_optimize(
        triangles: &mut Vec<Triangle>,
        bounds: &AABB,
        percent: Real,
        max_offset: Option<Real>,
        iterations: usize,
        b: &mut ::test::Bencher,
    ) {
        let mut bvh = BVH::build(triangles);
        let num_move = (triangles.len() as Real * percent) as usize;
        let mut seed = 0;

        for _ in 0..iterations {
            let updated =
                randomly_transform_scene(triangles, num_move, &bounds, max_offset, &mut seed);
            let updated: Vec<usize> = updated.into_iter().collect();
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

    #[bench]
    fn bench_intersect_120k_after_optimize_100p(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        intersect_scene_after_optimize(&mut triangles, &bounds, 1.0, None, 10, b);
    }

    /// Move `percent` `Triangle`s in the scene given by `triangles` `iterations` times.
    /// Afterwards optimize the `BVH` and benchmark the performance of intersecting this
    /// scene/`BVH`. Used to compare optimizing with rebuilding. For reference see
    /// `intersect_scene_after_optimize`.
    fn intersect_scene_with_rebuild(
        triangles: &mut Vec<Triangle>,
        bounds: &AABB,
        percent: Real,
        max_offset: Option<Real>,
        iterations: usize,
        b: &mut ::test::Bencher,
    ) {
        let num_move = (triangles.len() as Real * percent) as usize;
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

    #[bench]
    fn bench_intersect_120k_with_rebuild_100p(b: &mut ::test::Bencher) {
        let bounds = default_bounds();
        let mut triangles = create_n_cubes(10_000, &bounds);
        intersect_scene_with_rebuild(&mut triangles, &bounds, 1.0, None, 10, b);
    }

    /// Benchmark intersecting a `BVH` for Sponza after randomly moving one `Triangle` and
    /// optimizing.
    fn intersect_sponza_after_optimize(percent: Real, b: &mut ::test::Bencher) {
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

    #[bench]
    fn bench_intersect_sponza_after_optimize_100p(b: &mut ::test::Bencher) {
        intersect_sponza_after_optimize(1.0, b);
    }

    /// Benchmark intersecting a `BVH` for Sponza after rebuilding. Used to compare optimizing
    /// with rebuilding. For reference see `intersect_sponza_after_optimize`.
    fn intersect_sponza_with_rebuild(percent: Real, b: &mut ::test::Bencher) {
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

    #[bench]
    fn bench_intersect_sponza_with_rebuild_100p(b: &mut ::test::Bencher) {
        intersect_sponza_with_rebuild(1.0, b);
    }
}
