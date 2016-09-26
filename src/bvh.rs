use aabb::{AABB, Bounded};
use std::boxed::Box;
use std::f32;
use std::iter::repeat;

enum BVHNode {
    Leaf { aabb: AABB, shapes: Vec<usize> },
    Node {
        aabb: AABB,
        init: Box<BVHNode>,
        tail: Box<BVHNode>,
    },
}

impl BVHNode {
    pub fn new<T: Bounded>(shapes: &[T], indices: Vec<usize>) -> BVHNode {
        let (aabb_bounds, centroid_bounds) =
            indices.iter().fold((AABB::empty(), AABB::empty()),
                                |(aabb_bounds, centroid_bounds), idx| {
                                    let aabb = &shapes[*idx].aabb();
                                    (aabb_bounds.union_aabb(aabb),
                                     centroid_bounds.union_point(&aabb.center()))
                                });

        if indices.len() < 5 {
            return BVHNode::Leaf {
                aabb: aabb_bounds,
                shapes: indices,
            };
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
                self.aabb = self.aabb.union_aabb(aabb);
            }
        }

        /// Returns the union of two `Bucket`s.
        fn bucket_union(a: Bucket, b: &Bucket) -> Bucket {
            Bucket {
                size: a.size + b.size,
                aabb: a.aabb.union_aabb(&b.aabb),
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

        // Compute the costs for each configuration
        let costs = (0..11).map(|i| {
            let init = buckets.iter().take(i + 1).fold(Bucket::empty(), bucket_union);
            let tail = buckets.iter().skip(i + 1).fold(Bucket::empty(), bucket_union);

            0.125 +
            (init.size as f32 * init.aabb.surface_area() +
             tail.size as f32 * tail.aabb.surface_area()) / aabb_bounds.surface_area()
        });

        // Select the configuration with the minimal costs
        let (min_bucket, _) = costs.enumerate().fold((0, f32::INFINITY), |(min_bucket, min_cost),
                                                      (bucket_num, bucket_cost)| {
            if bucket_cost < min_cost {
                (bucket_num, bucket_cost)
            } else {
                (min_bucket, min_cost)
            }
        });

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
            aabb: aabb_bounds,
            init: Box::new(BVHNode::new(shapes, init_indices)),
            tail: Box::new(BVHNode::new(shapes, tail_indices)),
        }
    }

    fn print(&self, depth: usize) {
        let padding: String = repeat(" ").take(depth).collect();
        match *self {
            BVHNode::Node { ref aabb, ref init, ref tail } => {
                println!("{}AABB\t{:?}", padding, aabb);
                println!("{}init", padding);
                init.print(depth + 1);
                println!("{}tail", padding);
                tail.print(depth + 1);
            }
            BVHNode::Leaf { ref aabb, ref shapes } => {
                println!("{}AABB\t{:?}", padding, aabb);
                println!("{}shapes\t{:?}", padding, shapes);
            }
        }
    }
}

pub struct BVH<T: Bounded> {
    shapes: Vec<T>,
    root: BVHNode,
}

impl<T: Bounded> BVH<T> {
    pub fn new(shapes: Vec<T>) -> BVH<T> {
        let indices = (0..shapes.len()).collect::<Vec<usize>>();
        let root = BVHNode::new(&shapes, indices);
        BVH {
            shapes: shapes,
            root: root,
        }
    }

    pub fn print(&self) {
        self.root.print(0);
    }
}
