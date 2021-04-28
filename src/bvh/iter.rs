use crate::aabb::Bounded;
use crate::bvh::{BVH, BVHNode};
use crate::ray::Ray;

/// Iterator traverse a BVH without memory allocations
pub struct BVHIterator<'a, Shape: Bounded> {
    bvh: &'a BVH,
    ray: &'a Ray,
    shapes: &'a [Shape],
    stack: [usize; 32],
    node_index: usize,
    stack_size: usize,
    has_node: bool,
}

impl<'a, Shape: Bounded> BVHIterator<'a, Shape> {
    /// Creates a new BVHIterator
    pub fn new(bvh: &'a BVH, ray: &'a Ray, shapes: &'a [Shape]) -> Self {
        BVHIterator {
            /// Reference to the BVH to traverse
            bvh: bvh,

            /// Reference to the input ray
            ray: ray,

            /// Reference to the input shapes array
            shapes: shapes,
            
            /// Traversal stack. 4 billion items seems enough?
            stack: [0; 32],

            /// Position of the iterator in bvh.nodes
            node_index: 0,

            /// Size of the traversal stack
            stack_size: 0,

            /// Whether or not we have a valid node (or leaf)
            has_node: true,
        }
    }

    /// Test if stack is empty.
    fn is_stack_empty(&self) -> bool {
        return self.stack_size == 0;
    }

    /// Push node onto stack. Not guarded against overflow.
    fn stack_push(&mut self, node: usize) {
        self.stack[self.stack_size] = node;
        self.stack_size += 1;
    }

    /// Pop the stack and return the node. Not guarded against underflow.
    fn stack_pop(&mut self) -> usize {
        self.stack_size -= 1;
        return self.stack[self.stack_size];
    }

    /// Attempt to move to the left child of the current node.
    fn move_left(&mut self) {
        match self.bvh.nodes[self.node_index] {
            BVHNode::Node {
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
            BVHNode::Leaf { .. } => {
                self.has_node = false;
            }
        }
    }

    /// Attempt to move to the right child of the current node.
    fn move_right(&mut self) {
        match self.bvh.nodes[self.node_index] {
            BVHNode::Node {
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
            BVHNode::Leaf { .. } => {
                self.has_node = false;
            }
        }
    }
}

impl<'a, Shape: Bounded> Iterator for BVHIterator<'a, Shape> {
    type Item = &'a Shape;

    fn next(&mut self) -> Option<&'a Shape> {
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
                    BVHNode::Node { .. } => {
                        // If a node was pushed, now attempt to move to its right child.
                        self.move_right();
                    }
                    BVHNode::Leaf { shape_index, .. } => {
                        // We previously pushed a leaf node. This is the "visit" of the in-order traverse.
                        // Next time we call next we try to pop the stack again.
                        self.has_node = false;
                        return Some(&self.shapes[shape_index]);
                    }
                }
            }
        }
        return None;
    }
}

// Copy of part of the BH testing in testbase.
// TODO: Once iterators are part of the BoundingHierarchy trait we can move all this to testbase.
#[cfg(test)]
mod tests {
    use crate::bvh::{BVHNode, BVH};
    use std::collections::HashSet;
    use std::f32;
    
    use nalgebra::{Point3, Vector3};
    
    use crate::aabb::{Bounded, AABB};
    use crate::bounding_hierarchy::BHShape;
    use crate::ray::Ray;

    /// Define some `Bounded` structure.
    pub struct UnitBox {
        pub id: i32,
        pub pos: Point3<f32>,
        node_index: usize,
    }

    impl UnitBox {
        pub fn new(id: i32, pos: Point3<f32>) -> UnitBox {
            UnitBox {
                id: id,
                pos: pos,
                node_index: 0,
            }
        }
    }

    /// `UnitBox`'s `AABB`s are unit `AABB`s centered on the box's position.
    impl Bounded for UnitBox {
        fn aabb(&self) -> AABB {
            let min = self.pos + Vector3::new(-0.5, -0.5, -0.5);
            let max = self.pos + Vector3::new(0.5, 0.5, 0.5);
            AABB::with_bounds(min, max)
        }
    }

    impl BHShape for UnitBox {
        fn set_bh_node_index(&mut self, index: usize) {
            self.node_index = index;
        }

        fn bh_node_index(&self) -> usize {
            self.node_index
        }
    }

    /// Generate 21 `UnitBox`s along the X axis centered on whole numbers (-10,9,..,10).
    /// The index is set to the rounded x-coordinate of the box center.
    pub fn generate_aligned_boxes() -> Vec<UnitBox> {
        // Create 21 boxes along the x-axis
        let mut shapes = Vec::new();
        for x in -10..11 {
            shapes.push(UnitBox::new(x, Point3::new(x as f32, 0.0, 0.0)));
        }
        shapes
    }

    /// Creates a `BoundingHierarchy` for a fixed scene structure.
    pub fn build_some_bvh() -> (Vec<UnitBox>, BVH) {
        let mut boxes = generate_aligned_boxes();
        let bvh = BVH::build(&mut boxes);
        (boxes, bvh)
    }

    /// Given a ray, a bounding hierarchy, the complete list of shapes in the scene and a list of
    /// expected hits, verifies, whether the ray hits only the expected shapes.
    fn traverse_and_verify_vec(
        ray_origin: Point3<f32>,
        ray_direction: Vector3<f32>,
        all_shapes: &Vec<UnitBox>,
        bvh: &BVH,
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
        ray_origin: Point3<f32>,
        ray_direction: Vector3<f32>,
        all_shapes: &Vec<UnitBox>,
        bvh: &BVH,
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
        ray_origin: Point3<f32>,
        ray_direction: Vector3<f32>,
        all_shapes: &Vec<UnitBox>,
        bvh: &BVH,
        expected_shapes: &HashSet<i32>,
    ) {
        traverse_and_verify_vec(ray_origin, ray_direction, all_shapes, bvh, expected_shapes);
        traverse_and_verify_iterator(ray_origin, ray_direction, all_shapes, bvh, expected_shapes);
    }

    /// Perform some fixed intersection tests on BH structures.
    pub fn traverse_some_bvh() {
        let (all_shapes, bh) = build_some_bvh();

        {
            // Define a ray which traverses the x-axis from afar.
            let origin = Point3::new(-1000.0, 0.0, 0.0);
            let direction = Vector3::new(1.0, 0.0, 0.0);
            let mut expected_shapes = HashSet::new();

            // It should hit everything.
            for id in -10..11 {
                expected_shapes.insert(id);
            }
            traverse_and_verify_base(origin, direction, &all_shapes, &bh, &expected_shapes);
        }

        {
            // Define a ray which traverses the y-axis from afar.
            let origin = Point3::new(0.0, -1000.0, 0.0);
            let direction = Vector3::new(0.0, 1.0, 0.0);

            // It should hit only one box.
            let mut expected_shapes = HashSet::new();
            expected_shapes.insert(0);
            traverse_and_verify_base(origin, direction, &all_shapes, &bh, &expected_shapes);
        }

        {
            // Define a ray which intersects the x-axis diagonally.
            let origin = Point3::new(6.0, 0.5, 0.0);
            let direction = Vector3::new(-2.0, -1.0, 0.0);

            // It should hit exactly three boxes.
            let mut expected_shapes = HashSet::new();
            expected_shapes.insert(4);
            expected_shapes.insert(5);
            expected_shapes.insert(6);
            traverse_and_verify_base(origin, direction, &all_shapes, &bh, &expected_shapes);
        }
    }


    #[test]
    /// Runs some primitive tests for intersections of a ray with a fixed scene given as a BVH.
    fn test_traverse_bvh() {
        traverse_some_bvh();
    }

    #[test]
    /// Verify contents of the bounding hierarchy for a fixed scene structure
    fn test_bvh_shape_indices() {
        use std::collections::HashSet;

        let (all_shapes, bh) = build_some_bvh();

        // It should find all shape indices.
        let expected_shapes: HashSet<_> = (0..all_shapes.len()).collect();
        let mut found_shapes = HashSet::new();

        for node in bh.nodes.iter() {
            match *node {
                BVHNode::Node { .. } => {
                    assert_eq!(node.shape_index(), None);
                }
                BVHNode::Leaf { .. } => {
                    found_shapes.insert(
                        node.shape_index()
                            .expect("getting a shape index from a leaf node"),
                    );
                }
            }
        }

        assert_eq!(expected_shapes, found_shapes);
    }
}
