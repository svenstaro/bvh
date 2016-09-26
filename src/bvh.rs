use aabb::{AABB, Bounded};
use ray::Ray;
use std::boxed::Box;
use std::f32;
use std::iter::repeat;

/// Enum which describes the union type of `BVHNode`s.
enum BVHNode {
    /// Leaf node.
    Leaf {
        /// The shapes contained in this leaf.
        shapes: Vec<usize>,
    },
    /// Inner node.
    Node {
        /// The union `AABB` of the shapes in init.
        init_aabb: AABB,
        init: Box<BVHNode>,
        /// The union `AABB` of the shapes in tail.
        tail_aabb: AABB,
        tail: Box<BVHNode>,
    },
}

impl BVHNode {
    pub fn new<T: Bounded>(shapes: &[T], indices: Vec<usize>) -> BVHNode {

        // Helper function to accumulate the AABB union and the centroids AABB.
        fn grow_union_bounds(union_bounds: (AABB, AABB), shape_aabb: &AABB) -> (AABB, AABB) {
            let center = &shape_aabb.center();
            let union_aabbs = &union_bounds.0;
            let union_centroids = &union_bounds.1;
            (union_aabbs.union(shape_aabb), union_centroids.grow(center))
        }

        let mut union_bounds = Default::default();
        for index in &indices {
            union_bounds = grow_union_bounds(union_bounds, &shapes[*index].aabb());
        }
        let (aabb_bounds, centroid_bounds) = union_bounds;

        // If there are less than five elements, don't split anymore
        if indices.len() <= 5 {
            return BVHNode::Leaf { shapes: indices };
        }

        // Find the axis along which the shapes are spread the most
        let split_axis = centroid_bounds.largest_axis();
        let split_axis_size = centroid_bounds.max[split_axis] - centroid_bounds.min[split_axis];

        /// Defines a Bucket utility object
        #[derive(Copy, Clone)]
        struct Bucket {
            size: usize,
            aabb: AABB,
        }

        impl Bucket {
            /// Returns an empty bucket
            fn empty() -> Bucket {
                Bucket {
                    size: 0,
                    aabb: AABB::empty(),
                }
            }

            /// Extends this `Bucket` by the given `AABB`.
            fn add_aabb(&mut self, aabb: &AABB) {
                self.size += 1;
                self.aabb = self.aabb.union(aabb);
            }
        }

        /// Returns the union of two `Bucket`s.
        fn bucket_union(a: Bucket, b: &Bucket) -> Bucket {
            Bucket {
                size: a.size + b.size,
                aabb: a.aabb.union(&b.aabb),
            }
        }

        // Create twelve buckets, and twelve index assignment vectors
        let mut buckets = [Bucket::empty(); 12];
        let mut bucket_assignments: [Vec<usize>; 12] = Default::default();

        // Iterate through all shapes
        for idx in &indices {
            let shape = &shapes[*idx];
            let shape_aabb = shape.aabb();
            let shape_center = shape_aabb.center();

            // Get the relative position of the shape centroid [0.0..1.0]
            let bucket_num_relative = (shape_center[split_axis] - centroid_bounds.min[split_axis]) /
                                      split_axis_size;

            // Convert that to the actual `Bucket` number
            let bucket_num = (bucket_num_relative * 11.99) as usize;

            // Extend the selected `Bucket` and add the index to the actual bucket
            buckets[bucket_num].add_aabb(&shape_aabb);
            bucket_assignments[bucket_num].push(*idx);
        }

        // Compute the costs for each configuration and
        // select the configuration with the minimal costs
        let mut min_bucket = 0;
        let mut min_cost = f32::INFINITY;
        let mut init_aabb = AABB::empty();
        let mut tail_aabb = AABB::empty();
        for i in 0..11 {
            let init = buckets.iter().take(i + 1).fold(Bucket::empty(), bucket_union);
            let tail = buckets.iter().skip(i + 1).fold(Bucket::empty(), bucket_union);

            let cost = (init.size as f32 * init.aabb.surface_area() +
                        tail.size as f32 * tail.aabb.surface_area()) /
                       aabb_bounds.surface_area();

            if cost < min_cost {
                min_bucket = i;
                min_cost = cost;
                init_aabb = init.aabb;
                tail_aabb = tail.aabb;
            }
        }

        // Join together all index buckets, and proceed recursively
        let mut init_indices = Vec::new();
        for mut indices in bucket_assignments.iter_mut().take(min_bucket + 1) {
            init_indices.append(&mut indices);
        }
        let mut tail_indices = Vec::new();
        for mut indices in bucket_assignments.iter_mut().skip(min_bucket + 1) {
            tail_indices.append(&mut indices);
        }

        // Construct the actual data structure
        BVHNode::Node {
            init_aabb: init_aabb,
            init: Box::new(BVHNode::new(shapes, init_indices)),
            tail_aabb: tail_aabb,
            tail: Box::new(BVHNode::new(shapes, tail_indices)),
        }
    }

    fn print(&self, depth: usize) {
        let padding: String = repeat(" ").take(depth).collect();
        match *self {
            BVHNode::Node { ref init_aabb, ref init, ref tail_aabb, ref tail } => {
                println!("{}init", padding);
                init.print(depth + 1);
                println!("{}tail", padding);
                tail.print(depth + 1);
            }
            BVHNode::Leaf { ref shapes } => {
                println!("{}shapes\t{:?}", padding, shapes);
            }
        }
    }

    pub fn traverse_recursive(&self, ray: &Ray, indices: &mut Vec<usize>) {
        match *self {
            BVHNode::Node { ref init_aabb, ref init, ref tail_aabb, ref tail } => {
                if ray.intersects_aabb(init_aabb) {
                    init.traverse_recursive(ray, indices);
                }
                if ray.intersects_aabb(tail_aabb) {
                    tail.traverse_recursive(ray, indices);
                }
            }
            BVHNode::Leaf { ref shapes } => {
                for index in shapes {
                    indices.push(*index);
                }
            }
        }
    }
}

pub struct BVH {
    root: BVHNode,
}

impl BVH {
    pub fn new<T: Bounded>(shapes: Vec<T>) -> BVH {
        let indices = (0..shapes.len()).collect::<Vec<usize>>();
        let root = BVHNode::new(&shapes, indices);
        BVH { root: root }
    }

    pub fn print(&self) {
        self.root.print(0);
    }

    pub fn traverse_recursive<'a, T: Bounded>(&'a self, ray: &Ray, shapes: &'a [T]) -> Vec<&T> {
        let mut indices = Vec::new();
        self.root.traverse_recursive(ray, &mut indices);
        let mut hit_shapes = Vec::new();
        for index in &indices {
            let shape = &shapes[*index];
            if ray.intersects_aabb(&shape.aabb()) {
                hit_shapes.push(shape);
            }
        }
        hit_shapes
    }
}

#[cfg(test)]
mod tests {
    use aabb::AABB;
    use bvh::BVH;

    #[test]
    fn test_primitive_bvh() {}
}
