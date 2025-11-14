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
    use crate::aabb::{Aabb, Bounded};
    use crate::bounding_hierarchy::{BHShape, BHValue};
    use crate::bvh::Bvh;
    use crate::ray::Ray;
    use crate::testbase::{
        TAabb3, TBvh3, TPoint3, TRay3, TVector3, UnitBox, generate_aligned_boxes,
    };
    use alloc::vec::Vec;
    use std::collections::HashSet;

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
        let mut aabbs = [
            TAabb3 {
                min: TPoint3::new(-0.33333334, -5000.3335, -5000.3335),
                max: TPoint3::new(1.3333334, 0.33333334, 0.33333334),
            },
            TAabb3 {
                min: TPoint3::new(-5000.3335, -5000.3335, -5000.3335),
                max: TPoint3::new(0.33333334, 0.33333334, -4998.6665),
            },
            TAabb3 {
                min: TPoint3::new(-5000.3335, -5000.3335, -5000.3335),
                max: TPoint3::new(0.33333334, 0.33333334, 5000.3335),
            },
        ];
        let ray = TRay3::new(
            TPoint3::new(-5000.0, -5000.0, -5000.0),
            TVector3::new(1.0, 0.0, 0.0),
        );

        let bvh = TBvh3::build(&mut aabbs);
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
