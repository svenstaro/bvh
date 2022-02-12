use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::bvh::BVH;
use crate::{aabb::AABB, Real};

use super::BVHNode;

thread_local! {
    /// Thread local for doing a best first traversal of the bvh
    static HEAP: RefCell<Vec<BinaryHeap<BvhTraversalRes>>> = RefCell::new(Default::default());
}

/// Used to traverse the results from intersections
#[derive(Debug, Clone, Copy)]
pub struct BvhTraversalRes {
    /// Squared distance to min intersection
    pub t_min_squared: Real,
    /// bvh node to test next
    pub node_index: usize,
}

impl BvhTraversalRes {
    /// Create new instance of BvhTraversalRes
    pub fn new(node_index: usize, t_min: Real) -> Self {
        Self {
            node_index,
            t_min_squared: t_min,
        }
    }
}

impl Ord for BvhTraversalRes {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.t_min_squared
            .partial_cmp(&other.t_min_squared)
            .unwrap_or(Ordering::Equal)
            .reverse()
    }
}
impl PartialOrd for BvhTraversalRes {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for BvhTraversalRes {
    fn eq(&self, other: &Self) -> bool {
        self.node_index == other.node_index
    }
}

impl Eq for BvhTraversalRes {}

impl BVH {
    /// Walk BVH with the closest nodes first
    pub fn traverse_best_first_with_heap<Res>(
        &self,
        t_min: Real,
        t_max: Real,
        test_aabb: impl Fn(&AABB) -> Option<Real>,
        test_shape: impl Fn(usize) -> Option<(Real, Res)>,
        heap: &mut BinaryHeap<BvhTraversalRes>,
    ) -> Option<Res> {
        heap.clear();
        heap.push(BvhTraversalRes::new(0, 0.));

        let mut result = None;
        let mut curr_min = Real::INFINITY;

        while let Some(next) = heap.pop() {
            if curr_min < next.t_min_squared {
                break;
            }

            let node = self.nodes[next.node_index];
            match node {
                BVHNode::Leaf { shape_index, .. } => {
                    if let Some((dist, res)) = test_shape(shape_index) {
                        let dist_squared = dist * dist;
                        if dist_squared < curr_min && dist > t_min && dist < t_max {
                            curr_min = dist_squared;
                            result = Some(res);
                        }
                    }
                }
                BVHNode::Node {
                    child_l_index,
                    child_l_aabb,
                    child_r_index,
                    child_r_aabb,
                    ..
                } => {
                    if let Some(l_min) = test_aabb(&child_l_aabb) {
                        heap.push(BvhTraversalRes::new(child_l_index, l_min))
                    }
                    if let Some(r_min) = test_aabb(&child_r_aabb) {
                        heap.push(BvhTraversalRes::new(child_r_index, r_min))
                    }
                }
            }
        }
        result
    }

    /// Traverses best first using a thread local heap
    pub fn traverse_best_first<Res>(
        &self,
        t_min: Real,
        t_max: Real,
        test_aabb: impl Fn(&AABB) -> Option<Real>,
        test_shape: impl Fn(usize) -> Option<(Real, Res)>,
    ) -> Option<Res> {
        let mut heap = HEAP.with(|h| {
            if let Some(x) = h.borrow_mut().pop() {
                x
            } else {
                Default::default()
            }
        });
        let res =
            self.traverse_best_first_with_heap(t_min, t_max, test_aabb, test_shape, &mut heap);

        HEAP.with(|h| h.borrow_mut().push(heap));
        res
    }
}
