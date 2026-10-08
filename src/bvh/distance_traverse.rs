use alloc::collections::BinaryHeap;
use core::cmp::Ordering;

use crate::aabb::{Aabb, Bounded};
use crate::bounding_hierarchy::BHValue;
use crate::bvh::{Bvh, BvhNode, iter_initially_has_node};
use crate::ray::Ray;

#[derive(Debug, Clone, Copy)]
struct DistNodePair<T: PartialOrd> {
    dist: T,
    node_index: usize,
}

impl<T: PartialOrd> Eq for DistNodePair<T> {}

impl<T: PartialOrd> PartialEq<Self> for DistNodePair<T> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<T: PartialOrd> PartialOrd<Self> for DistNodePair<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PartialOrd> Ord for DistNodePair<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap()
    }
}

/// Iterator to traverse a [`Bvh`] in order from nearest [`Aabb`] to farthest for [`Ray`],
/// or vice versa, without memory allocations.
///
/// This is a best-effort iterator that orders interior parent nodes before ordering child
/// nodes, so the output is not necessarily perfectly sorted.
pub struct DistanceTraverseIterator<
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
    /// Traversal heap. Store distances and nodes
    heap: BinaryHeap<DistNodePair<T>>,
}

impl<'bvh, 'shape, T, const D: usize, Shape: Bounded<T, D>, const ASCENDING: bool>
    DistanceTraverseIterator<'bvh, 'shape, T, D, Shape, ASCENDING>
where
    T: BHValue,
{
    /// Creates a new [`DistanceTraverseIterator `]
    pub fn new(bvh: &'bvh Bvh<T, D>, ray: &'bvh Ray<T, D>, shapes: &'shape [Shape]) -> Self {
        let mut iterator = DistanceTraverseIterator {
            bvh,
            ray,
            shapes,
            heap: BinaryHeap::new(),
        };

        if iter_initially_has_node(bvh, ray, shapes) {
            // init starting node. Distance doesn't matter
            iterator.add_to_heap(T::zero(), 0);
        }

        iterator
    }

    /// Unpack node.
    /// If it is a leaf returns shape index, else - add childs to heap
    fn unpack_node(&mut self, node_index: usize) -> Option<usize> {
        match self.bvh.nodes[node_index] {
            BvhNode::Node {
                child_l_index,
                ref child_l_aabb,
                child_r_index,
                ref child_r_aabb,
                ..
            } => {
                self.process_child_intersection(child_l_index, child_l_aabb);
                self.process_child_intersection(child_r_index, child_r_aabb);
                None
            }
            BvhNode::Leaf { shape_index, .. } => Some(shape_index),
        }
    }

    /// Intersect child node with a ray and add it to the heap.
    fn process_child_intersection(&mut self, child_node_index: usize, child_aabb: &Aabb<T, D>) {
        let dists_opt = self.ray.intersection_slice_for_aabb(child_aabb);

        // if there is an intersection
        if let Some((dist_to_entry_point, dist_to_exit_point)) = dists_opt {
            let dist_to_compare = if ASCENDING {
                // cause our iterator from nearest to farthest shapes,
                // we compare them by first intersection point - entry point
                dist_to_entry_point
            } else {
                // cause our iterator from farthest to nearest shapes,
                // we compare them by second intersection point - exit point
                dist_to_exit_point
            };
            self.add_to_heap(dist_to_compare, child_node_index);
        };
    }

    fn add_to_heap(&mut self, dist_to_node: T, node_index: usize) {
        // cause we use max-heap, it store max value on the top
        if ASCENDING {
            // we need the smallest distance, so we negate value
            self.heap.push(DistNodePair {
                dist: dist_to_node.neg(),
                node_index,
            });
        } else {
            // we need the biggest distance, so everything fine
            self.heap.push(DistNodePair {
                dist: dist_to_node,
                node_index,
            });
        };
    }
}

impl<'shape, T, const D: usize, Shape: Bounded<T, D>, const ASCENDING: bool> Iterator
    for DistanceTraverseIterator<'_, 'shape, T, D, Shape, ASCENDING>
where
    T: BHValue,
{
    type Item = &'shape Shape;

    fn next(&mut self) -> Option<&'shape Shape> {
        while let Some(heap_leader) = self.heap.pop() {
            // Get favorite (nearest/farthest) node and unpack
            let DistNodePair {
                dist: _,
                node_index,
            } = heap_leader;

            if let Some(shape_index) = self.unpack_node(node_index) {
                // unpacked leaf
                return Some(&self.shapes[shape_index]);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounding_hierarchy::BHShape;
    use nalgebra::{Point3, Vector3};

    // Test-only: treat an `Aabb` as a shape so we can build a BVH of plain boxes.
    impl<T: BHValue, const D: usize> BHShape<T, D> for Aabb<T, D> {
        fn bh_node_index(&self) -> usize {
            unimplemented!();
        }

        fn set_bh_node_index(&mut self, _: usize) {
            // No-op.
        }
    }

    #[test]
    fn test_overlapping_child_order() {
        let point = |x, y, z| Point3::new(x, y, z);
        let mut aabbs = [
            Aabb {
                min: point(-0.33333334, -5000.3335, -5000.3335),
                max: point(1.3333334, 0.33333334, 0.33333334),
            },
            Aabb {
                min: point(-5000.3335, -5000.3335, -5000.3335),
                max: point(0.33333334, 0.33333334, -4998.6665),
            },
            Aabb {
                min: point(-5000.3335, -5000.3335, -5000.3335),
                max: point(0.33333334, 0.33333334, 5000.3335),
            },
        ];
        let ray = Ray::new(
            point(-5000.0, -5000.0, -5000.0),
            Vector3::new(1.0, 0.0, 0.0),
        );

        let bvh = Bvh::<f32, 3>::build(&mut aabbs);
        assert!(
            bvh.nearest_traverse_iterator(&ray, &aabbs)
                .is_sorted_by(|a, b| {
                    let (a, _) = ray.intersection_slice_for_aabb(a).unwrap();
                    let (b, _) = ray.intersection_slice_for_aabb(b).unwrap();
                    a <= b
                })
        );
    }
}
