use nalgebra::{ClosedMul, ClosedSub, Scalar, SimdPartialOrd};
use num::Zero;

use crate::aabb::Bounded;
use crate::bvh::{Bvh, BvhNode};
use crate::ray::Ray;

/// Iterator to traverse a [`Bvh`] without memory allocations
pub struct BvhTraverseIterator<'bvh, 'shape, T: Scalar + Copy, const D: usize, Shape: Bounded<T, D>>
{
    /// Reference to the Bvh to traverse
    bvh: &'bvh Bvh<T, D>,
    /// Reference to the input ray
    ray: &'bvh Ray<T, D>,
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

impl<'bvh, 'shape, T, const D: usize, Shape: Bounded<T, D>>
    BvhTraverseIterator<'bvh, 'shape, T, D, Shape>
where
    T: Scalar + Copy + SimdPartialOrd + ClosedSub + PartialOrd + ClosedMul + Zero,
{
    /// Creates a new `BvhTraverseIterator`
    pub fn new(bvh: &'bvh Bvh<T, D>, ray: &'bvh Ray<T, D>, shapes: &'shape [Shape]) -> Self {
        BvhTraverseIterator {
            bvh,
            ray,
            shapes,
            stack: [0; 32],
            node_index: 0,
            stack_size: 0,
            has_node: true,
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
    /// If it is a leaf, or the ray does not intersect the node `Aabb`, `has_node` will become false.
    fn move_left(&mut self) {
        match self.bvh.nodes[self.node_index] {
            BvhNode::Node {
                child_l_index,
                ref child_l_aabb,
                ..
            } => {
                if self.ray.intersects_aabb(child_l_aabb) {
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
    /// If it is a leaf, or the ray does not intersect the node `Aabb`, `has_node` will become false.
    fn move_right(&mut self) {
        match self.bvh.nodes[self.node_index] {
            BvhNode::Node {
                child_r_index,
                ref child_r_aabb,
                ..
            } => {
                if self.ray.intersects_aabb(child_r_aabb) {
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

impl<'bvh, 'shape, T, const D: usize, Shape: Bounded<T, D>> Iterator
    for BvhTraverseIterator<'bvh, 'shape, T, D, Shape>
where
    T: Scalar + Copy + SimdPartialOrd + ClosedSub + PartialOrd + ClosedMul + Zero,
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

// Copy of part of the BH testing in testbase.
// TODO: Once iterators are part of the BoundingHierarchy trait we can move all this to testbase.
#[cfg(test)]
mod tests {
    use crate::ray::Ray;
    use crate::testbase::{generate_aligned_boxes, TBvh3, TPoint3, TVector3, UnitBox};
    use std::collections::HashSet;

    /// Creates a `Bvh` for a fixed scene structure.
    pub fn build_some_bvh() -> (Vec<UnitBox>, TBvh3) {
        let mut boxes = generate_aligned_boxes();
        let bvh = TBvh3::build(&mut boxes);
        (boxes, bvh)
    }

    /// Given a ray, a bounding hierarchy, the complete list of shapes in the scene and a list of
    /// expected hits, verifies, whether the ray hits only the expected shapes.
    fn traverse_and_verify_vec(
        ray_origin: TPoint3,
        ray_direction: TVector3,
        all_shapes: &[UnitBox],
        bvh: &TBvh3,
        expected_shapes: &HashSet<i32>,
    ) {
        let ray = Ray::new(ray_origin, ray_direction);
        let hit_shapes = bvh.traverse(&ray, all_shapes);

        assert_eq!(expected_shapes.len(), hit_shapes.len());
        for shape in hit_shapes {
            assert!(expected_shapes.contains(&shape.id));
        }
    }

    fn traverse_and_verify_iterator(
        ray_origin: TPoint3,
        ray_direction: TVector3,
        all_shapes: &[UnitBox],
        bvh: &TBvh3,
        expected_shapes: &HashSet<i32>,
    ) {
        let ray = Ray::new(ray_origin, ray_direction);
        let it = bvh.traverse_iterator(&ray, all_shapes);

        let mut count = 0;
        for shape in it {
            assert!(expected_shapes.contains(&shape.id));
            count += 1;
        }
        assert_eq!(expected_shapes.len(), count);
    }

    fn traverse_and_verify_base(
        ray_origin: TPoint3,
        ray_direction: TVector3,
        all_shapes: &[UnitBox],
        bvh: &TBvh3,
        expected_shapes: &HashSet<i32>,
    ) {
        traverse_and_verify_vec(ray_origin, ray_direction, all_shapes, bvh, expected_shapes);
        traverse_and_verify_iterator(ray_origin, ray_direction, all_shapes, bvh, expected_shapes);
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
            traverse_and_verify_base(origin, direction, &all_shapes, &bvh, &expected_shapes);
        }

        {
            // Define a ray which traverses the y-axis from afar.
            let origin = TPoint3::new(0.0, -1000.0, 0.0);
            let direction = TVector3::new(0.0, 1.0, 0.0);

            // It should hit only one box.
            let mut expected_shapes = HashSet::new();
            expected_shapes.insert(0);
            traverse_and_verify_base(origin, direction, &all_shapes, &bvh, &expected_shapes);
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
            traverse_and_verify_base(origin, direction, &all_shapes, &bvh, &expected_shapes);
        }
    }

    #[test]
    /// Runs some primitive tests for intersections of a ray with a fixed scene given as a Bvh.
    fn test_traverse_bvh() {
        traverse_some_bvh();
    }
}

#[cfg(all(feature = "bench", test))]
mod bench {
    use crate::testbase::{create_ray, load_sponza_scene, TBvh3};

    #[bench]
    /// Benchmark the traversal of a `Bvh` with the Sponza scene with Vec return.
    fn bench_intersect_128rays_sponza_vec(b: &mut ::test::Bencher) {
        let (mut triangles, bounds) = load_sponza_scene();
        let bvh = TBvh3::build(&mut triangles);

        let mut seed = 0;
        b.iter(|| {
            for _ in 0..128 {
                let ray = create_ray(&mut seed, &bounds);

                // Traverse the `Bvh` recursively.
                let hits = bvh.traverse(&ray, &triangles);

                // Traverse the resulting list of positive `Aabb` tests
                for triangle in &hits {
                    ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c);
                }
            }
        });
    }

    #[bench]
    /// Benchmark the traversal of a `Bvh` with the Sponza scene with `BvhTraverseIterator`.
    fn bench_intersect_128rays_sponza_iter(b: &mut ::test::Bencher) {
        let (mut triangles, bounds) = load_sponza_scene();
        let bvh = TBvh3::build(&mut triangles);

        let mut seed = 0;
        b.iter(|| {
            for _ in 0..128 {
                let ray = create_ray(&mut seed, &bounds);

                // Traverse the `Bvh` recursively.
                let hits = bvh.traverse_iterator(&ray, &triangles);

                // Traverse the resulting list of positive `Aabb` tests
                for triangle in hits {
                    ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c);
                }
            }
        });
    }
}
