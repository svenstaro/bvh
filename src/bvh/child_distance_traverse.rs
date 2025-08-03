use crate::aabb::Bounded;
use crate::bounding_hierarchy::BHValue;
use crate::bvh::{iter_initially_has_node, Bvh, BvhNode};
use crate::ray::Ray;

#[derive(Debug, Clone, Copy)]
enum RestChild {
    Left,
    Right,
    None,
}

/// Iterator to traverse a [`Bvh`] in order from nearest [`Aabb`] to farthest for [`Ray`],
/// or vice versa, without memory allocations.
///
/// This is a best-effort iterator that orders interior parent nodes before ordering child
/// nodes, so the output is not necessarily perfectly sorted.
pub struct ChildDistanceTraverseIterator<
    'bvh,
    'shape,
    T: BHValue,
    const D: usize,
    Shape: Bounded<T, D>,
    const ASCENDING: bool,
> {
    /// Reference to the Bvh to traverse
    bvh: &'bvh Bvh<T, D>,
    /// Reference to the input ray
    ray: &'bvh Ray<T, D>,
    /// Reference to the input shapes array
    shapes: &'shape [Shape],
    /// Traversal stack. 4 billion items seems enough?
    stack: [(usize, RestChild); 32],
    /// Position of the iterator in bvh.nodes
    node_index: usize,
    /// Size of the traversal stack
    stack_size: usize,
    /// Whether or not we have a valid node (or leaf)
    has_node: bool,
}

impl<'bvh, 'shape, T, const D: usize, Shape: Bounded<T, D>, const ASCENDING: bool>
    ChildDistanceTraverseIterator<'bvh, 'shape, T, D, Shape, ASCENDING>
where
    T: BHValue,
{
    /// Creates a new [`DistanceTraverseIterator`]
    pub fn new(bvh: &'bvh Bvh<T, D>, ray: &'bvh Ray<T, D>, shapes: &'shape [Shape]) -> Self {
        ChildDistanceTraverseIterator {
            bvh,
            ray,
            shapes,
            stack: [(0, RestChild::None); 32],
            node_index: 0,
            stack_size: 0,
            has_node: iter_initially_has_node(bvh, ray, shapes),
        }
    }

    /// Return `true` if stack is empty.
    fn is_stack_empty(&self) -> bool {
        self.stack_size == 0
    }

    /// Push node onto stack.
    ///
    /// # Panics
    ///
    /// Panics if `stack[stack_size]` is out of bounds.
    fn stack_push(&mut self, nodes: (usize, RestChild)) {
        self.stack[self.stack_size] = nodes;
        self.stack_size += 1;
    }

    /// Pop the stack and return the node.
    ///
    /// # Panics
    ///
    /// Panics if `stack_size` underflows.
    fn stack_pop(&mut self) -> (usize, RestChild) {
        self.stack_size -= 1;
        self.stack[self.stack_size]
    }

    /// Attempt to move to the child node that is closest/furthest to the ray, depending on `ASCENDING`,
    /// relative to the current node.
    /// If it is a leaf, or the [`Ray`] does not intersect the any node [`Aabb`], `has_node` will become `false`.
    fn move_first_priority(&mut self) -> (usize, RestChild) {
        let current_node_index = self.node_index;
        match self.bvh.nodes[current_node_index] {
            BvhNode::Node {
                child_l_index,
                ref child_l_aabb,
                child_r_index,
                ref child_r_aabb,
                ..
            } => {
                let left_dist = self
                    .ray
                    .intersection_slice_for_aabb(child_l_aabb)
                    .map(|(d, _)| d);
                let right_dist = self
                    .ray
                    .intersection_slice_for_aabb(child_r_aabb)
                    .map(|(d, _)| d);

                match (left_dist, right_dist) {
                    (None, None) => {
                        // no intersections with any children
                        self.has_node = false;
                        (current_node_index, RestChild::None)
                    }
                    (Some(_), None) => {
                        // has intersection only with left child
                        self.has_node = true;
                        self.node_index = child_l_index;
                        let rest_child = RestChild::None;
                        (current_node_index, rest_child)
                    }
                    (None, Some(_)) => {
                        // has intersection only with right child
                        self.has_node = true;
                        self.node_index = child_r_index;
                        let rest_child = RestChild::None;
                        (current_node_index, rest_child)
                    }
                    (Some(left_dist), Some(right_dist)) => {
                        if (left_dist > right_dist) ^ !ASCENDING {
                            // right is higher priority than left
                            self.has_node = true;
                            self.node_index = child_r_index;
                            let rest_child = RestChild::Left;
                            (current_node_index, rest_child)
                        } else {
                            // left is higher priority than right
                            self.has_node = true;
                            self.node_index = child_l_index;
                            let rest_child = RestChild::Right;
                            (current_node_index, rest_child)
                        }
                    }
                }
            }
            BvhNode::Leaf { .. } => {
                self.has_node = false;
                (current_node_index, RestChild::None)
            }
        }
    }

    /// Attempt to move to the rest not visited child of the current node.
    /// If it is a leaf, or the [`Ray`] does not intersect the node [`Aabb`], `has_node` will become `false`.
    fn move_rest(&mut self, rest_child: RestChild) {
        match self.bvh.nodes[self.node_index] {
            BvhNode::Node {
                child_r_index,
                child_l_index,
                ..
            } => match rest_child {
                RestChild::Left => {
                    self.node_index = child_l_index;
                    self.has_node = true;
                }
                RestChild::Right => {
                    self.node_index = child_r_index;
                    self.has_node = true;
                }
                RestChild::None => {
                    self.has_node = false;
                }
            },
            BvhNode::Leaf { .. } => {
                self.has_node = false;
            }
        }
    }
}

impl<'shape, T, const D: usize, Shape: Bounded<T, D>, const ASCENDING: bool> Iterator
    for ChildDistanceTraverseIterator<'_, 'shape, T, D, Shape, ASCENDING>
where
    T: BHValue,
{
    type Item = &'shape Shape;

    fn next(&mut self) -> Option<&'shape Shape> {
        loop {
            if self.is_stack_empty() && !self.has_node {
                // Completed traversal.
                break;
            }

            if self.has_node {
                // If we have any node, attempt to move to its nearest child.
                let stack_info = self.move_first_priority();
                // Save current node and farthest child
                self.stack_push(stack_info)
            } else {
                // Go back up the stack and see if a node or leaf was pushed.
                let (node_index, rest_child) = self.stack_pop();
                self.node_index = node_index;
                match self.bvh.nodes[self.node_index] {
                    BvhNode::Node { .. } => {
                        // If a node was pushed, now move to `unvisited` rest child, next in order.
                        self.move_rest(rest_child);
                    }
                    BvhNode::Leaf { shape_index, .. } => {
                        // We previously pushed a leaf node. This is the "visit" of the in-order traverse.
                        // Next time we call `next()` we try to pop the stack again.
                        return Some(&self.shapes[shape_index]);
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::aabb::Bounded;
    use crate::bvh::Bvh;
    use crate::ray::Ray;
    use crate::testbase::{generate_aligned_boxes, TBvh3, TPoint3, TVector3, UnitBox};
    use alloc::vec::Vec;
    use hashbrown::HashSet;

    /// Create a `Bvh` for a fixed scene structure.
    pub fn build_some_bvh() -> (Vec<UnitBox>, TBvh3) {
        let mut boxes = generate_aligned_boxes();
        let bvh = Bvh::build(&mut boxes);
        (boxes, bvh)
    }

    /// Create a `Bvh` for an empty scene structure.
    pub fn build_empty_bvh() -> (Vec<UnitBox>, TBvh3) {
        let mut boxes = Vec::new();
        let bvh = Bvh::build(&mut boxes);
        (boxes, bvh)
    }

    fn traverse_distance_and_verify_order(
        ray_origin: TPoint3,
        ray_direction: TVector3,
        all_shapes: &[UnitBox],
        bvh: &TBvh3,
        expected_shapes: &HashSet<i32>,
    ) {
        let ray = Ray::new(ray_origin, ray_direction);
        let near_it = bvh.nearest_traverse_iterator(&ray, all_shapes);
        let far_it = bvh.farthest_traverse_iterator(&ray, all_shapes);

        let mut count = 0;
        let mut prev_near_dist = -1.0;
        let mut prev_far_dist = f32::INFINITY;

        for (near_shape, far_shape) in near_it.zip(far_it) {
            let (intersect_near_dist, _) =
                ray.intersection_slice_for_aabb(&near_shape.aabb()).unwrap();
            let (intersect_far_dist, _) =
                ray.intersection_slice_for_aabb(&far_shape.aabb()).unwrap();

            assert!(expected_shapes.contains(&near_shape.id));
            assert!(expected_shapes.contains(&far_shape.id));
            assert!(prev_near_dist <= intersect_near_dist);
            assert!(prev_far_dist >= intersect_far_dist);

            count += 1;
            prev_near_dist = intersect_near_dist;
            prev_far_dist = intersect_far_dist;
        }
        assert_eq!(expected_shapes.len(), count);
    }

    /// Perform some fixed intersection tests on BH structures.
    pub fn traverse_some_bvh() {
        let (all_shapes, bvh) = build_some_bvh();

        {
            // Define a ray which traverses the x-axis from afar.
            let origin = TPoint3::new(-1000.0, 0.0, 0.0);
            let direction = TVector3::new(1.0, 0.0, 0.0);
            let mut expected_shapes = HashSet::new();

            // It should hit everything.
            for id in -10..11 {
                expected_shapes.insert(id);
            }
            traverse_distance_and_verify_order(
                origin,
                direction,
                &all_shapes,
                &bvh,
                &expected_shapes,
            );
        }

        {
            // Define a ray which intersects the x-axis diagonally.
            let origin = TPoint3::new(6.0, 0.5, 0.0);
            let direction = TVector3::new(-2.0, -1.0, 0.0);

            // It should hit exactly three boxes.
            let mut expected_shapes = HashSet::new();
            expected_shapes.insert(4);
            expected_shapes.insert(5);
            expected_shapes.insert(6);
            traverse_distance_and_verify_order(
                origin,
                direction,
                &all_shapes,
                &bvh,
                &expected_shapes,
            );
        }
    }

    #[test]
    /// Runs some primitive tests for intersections of a ray with a fixed scene given as a Bvh.
    fn test_traverse_bvh() {
        traverse_some_bvh();
    }

    #[test]
    fn test_traverse_empty_bvh() {
        let (shapes, bvh) = build_empty_bvh();

        // Define an arbitrary ray.
        let origin = TPoint3::new(0.0, 0.0, 0.0);
        let direction = TVector3::new(1.0, 0.0, 0.0);
        let ray = Ray::new(origin, direction);

        // Ensure distance traversal doesn't panic.
        assert_eq!(bvh.nearest_traverse_iterator(&ray, &shapes).count(), 0);
        assert_eq!(bvh.farthest_traverse_iterator(&ray, &shapes).count(), 0);
    }
}
