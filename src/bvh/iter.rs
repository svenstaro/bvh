use crate::aabb::{Bounded, IntersectsAabb};
use crate::bounding_hierarchy::BHValue;
use crate::bvh::{Bvh, BvhNode};

/// Iterator to traverse a [`Bvh`] without memory allocations
pub struct BvhTraverseIterator<
    'bvh,
    'shape,
    T: BHValue,
    const D: usize,
    Query: IntersectsAabb<T, D>,
    Shape: Bounded<T, D>,
> {
    /// Reference to the [`Bvh`] to traverse
    bvh: &'bvh Bvh<T, D>,
    /// Reference to the input query
    query: &'bvh Query,
    /// Reference to the input shapes array
    shapes: &'shape [Shape],
    /// Traversal stack. 4 billion items seems enough?
    stack: [usize; 32],
    /// Position of the iterator in bvh.nodes
    node_index: usize,
    /// Size of the traversal stack
    stack_size: usize,
    /// Whether or not we have a valid node (or leaf)
    has_node: bool,
}

impl<'bvh, 'shape, T: BHValue, const D: usize, Query: IntersectsAabb<T, D>, Shape: Bounded<T, D>>
    BvhTraverseIterator<'bvh, 'shape, T, D, Query, Shape>
{
    /// Creates a new [`BvhTraverseIterator`]
    pub fn new(bvh: &'bvh Bvh<T, D>, query: &'bvh Query, shapes: &'shape [Shape]) -> Self {
        BvhTraverseIterator {
            bvh,
            query,
            shapes,
            stack: [0; 32],
            node_index: 0,
            stack_size: 0,
            has_node: iter_initially_has_node(bvh, query, shapes),
        }
    }

    /// Test if stack is empty.
    fn is_stack_empty(&self) -> bool {
        self.stack_size == 0
    }

    /// Push node onto stack.
    ///
    /// # Panics
    ///
    /// Panics if `stack[stack_size]` is out of bounds.
    fn stack_push(&mut self, node: usize) {
        self.stack[self.stack_size] = node;
        self.stack_size += 1;
    }

    /// Pop the stack and return the node.
    ///
    /// # Panics
    ///
    /// Panics if `stack_size` underflows.
    fn stack_pop(&mut self) -> usize {
        self.stack_size -= 1;
        self.stack[self.stack_size]
    }

    /// Attempt to move to the left node child of the current node.
    /// If it is a leaf, or the ray does not intersect the node [`Aabb`], `has_node` will become false.
    fn move_left(&mut self) {
        match self.bvh.nodes[self.node_index] {
            BvhNode::Node {
                child_l_index,
                ref child_l_aabb,
                ..
            } => {
                if self.query.intersects_aabb(child_l_aabb) {
                    self.node_index = child_l_index;
                    self.has_node = true;
                } else {
                    self.has_node = false;
                }
            }
            BvhNode::Leaf { .. } => {
                self.has_node = false;
            }
        }
    }

    /// Attempt to move to the right node child of the current node.
    /// If it is a leaf, or the ray does not intersect the node [`Aabb`], `has_node` will become false.
    fn move_right(&mut self) {
        match self.bvh.nodes[self.node_index] {
            BvhNode::Node {
                child_r_index,
                ref child_r_aabb,
                ..
            } => {
                if self.query.intersects_aabb(child_r_aabb) {
                    self.node_index = child_r_index;
                    self.has_node = true;
                } else {
                    self.has_node = false;
                }
            }
            BvhNode::Leaf { .. } => {
                self.has_node = false;
            }
        }
    }
}

impl<'shape, T: BHValue, const D: usize, Query: IntersectsAabb<T, D>, Shape: Bounded<T, D>> Iterator
    for BvhTraverseIterator<'_, 'shape, T, D, Query, Shape>
{
    type Item = &'shape Shape;

    fn next(&mut self) -> Option<&'shape Shape> {
        loop {
            if self.is_stack_empty() && !self.has_node {
                // Completed traversal.
                break;
            }
            if self.has_node {
                // If we have any node, save it and attempt to move to its left child.
                self.stack_push(self.node_index);
                self.move_left();
            } else {
                // Go back up the stack and see if a node or leaf was pushed.
                self.node_index = self.stack_pop();
                match self.bvh.nodes[self.node_index] {
                    BvhNode::Node { .. } => {
                        // If a node was pushed, now attempt to move to its right child.
                        self.move_right();
                    }
                    BvhNode::Leaf { shape_index, .. } => {
                        // We previously pushed a leaf node. This is the "visit" of the in-order traverse.
                        // Next time we call `next()` we try to pop the stack again.
                        self.has_node = false;
                        return Some(&self.shapes[shape_index]);
                    }
                }
            }
        }
        None
    }
}

/// This computation is common to all `Bvh` traversal iterators.
///
/// It is designed to handle two extraordinary cases:
/// 1. Empty BVH. This case requires an empty iterator. This is accomplished by setting `has_node`
///    to `false` to indicate the absence of any nodes.
/// 2. Single-node BVH, in which the root node is a leaf node. This case requires returning the
///    root shape, if and only if its AABB is intersected by the ray. This is accomplished by
///    setting `has_node` based on manually checking intersection.
///
/// Finally, if the root is an interior node, that is the normal case. We set `has_node` to true so
/// the iterator can visit the root node and decide what to do next based on the root node's child
/// AABB's.
pub(crate) fn iter_initially_has_node<
    T: BHValue,
    const D: usize,
    Query: IntersectsAabb<T, D>,
    Shape: Bounded<T, D>,
>(
    bvh: &Bvh<T, D>,
    query: &Query,
    shapes: &[Shape],
) -> bool {
    match bvh.nodes.first() {
        // Only process the root leaf node if the shape's AABB is intersected.
        Some(BvhNode::Leaf { shape_index, .. }) => {
            query.intersects_aabb(&shapes[*shape_index].aabb())
        }
        Some(_) => true,
        None => false,
    }
}
