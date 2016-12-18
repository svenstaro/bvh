//! This module defines a [`BVH`] building procedure as well as a [`BVH`] flattening procedure
//! so that the recursive structure can be easily used in compute shaders.
//!
//! [`BVH`]: struct.BVH.html
//!

use EPSILON;
use aabb::{AABB, Bounded};
use ray::Ray;
use std::boxed::Box;
use std::f32;
use std::iter::repeat;

/// Enum which describes the union type of a node in a [`BVH`].
/// This structure does not allow for storing a root node's [`AABB`]. Therefore rays
/// which do not hit the root [`AABB`] perform two [`AABB`] tests (left/right) instead of one.
/// On the other hand this structure decreases the total number of indirections when traversing
/// the BVH. Only those nodes are accessed, which are definetely hit.
///
/// [`AABB`]: ../aabb/struct.AABB.html
/// [`BVH`]: struct.BVH.html
///
pub enum BVHNode {
    /// Leaf node.
    Leaf {
        /// The shapes contained in this leaf.
        shapes: Vec<usize>,
    },
    /// Inner node.
    Node {
        /// The convex hull of the shapes' `AABB`s in child_l.
        child_l_aabb: AABB,

        /// Left subtree.
        child_l: Box<BVHNode>,

        /// The convex hull of the shapes' `AABB`s in child_r.
        child_r_aabb: AABB,

        /// Right subtree.
        child_r: Box<BVHNode>,
    },
}

impl BVHNode {
    /// Builds a [`BVHNode`] recursively using SAH partitioning.
    ///
    /// [`BVHNode`]: enum.BVHNode.html
    ///
    pub fn build<T: Bounded>(shapes: &[T], indices: Vec<usize>) -> BVHNode {
        // Helper function to accumulate the AABB joint and the centroids AABB
        fn grow_convex_hull(convex_hull: (AABB, AABB), shape_aabb: &AABB) -> (AABB, AABB) {
            let center = &shape_aabb.center();
            let convex_hull_aabbs = &convex_hull.0;
            let convex_hull_centroids = &convex_hull.1;
            (convex_hull_aabbs.join(shape_aabb), convex_hull_centroids.grow(center))
        }

        let mut convex_hull = Default::default();
        for index in &indices {
            convex_hull = grow_convex_hull(convex_hull, &shapes[*index].aabb());
        }
        let (aabb_bounds, centroid_bounds) = convex_hull;

        // If there are five or fewer elements, don't split anymore
        if indices.len() <= 5 {
            return BVHNode::Leaf { shapes: indices };
        }

        // Find the axis along which the shapes are spread the most
        let split_axis = centroid_bounds.largest_axis();
        let split_axis_size = centroid_bounds.max[split_axis] - centroid_bounds.min[split_axis];

        if split_axis_size < EPSILON {
            return BVHNode::Leaf { shapes: indices };
        }

        /// Defines a Bucket utility object.
        #[derive(Copy, Clone)]
        struct Bucket {
            size: usize,
            aabb: AABB,
        }

        impl Bucket {
            /// Returns an empty bucket.
            fn empty() -> Bucket {
                Bucket {
                    size: 0,
                    aabb: AABB::empty(),
                }
            }

            /// Extends this `Bucket` by the given `AABB`.
            fn add_aabb(&mut self, aabb: &AABB) {
                self.size += 1;
                self.aabb = self.aabb.join(aabb);
            }
        }

        /// Joins two `Bucket`s.
        fn join_bucket(a: Bucket, b: &Bucket) -> Bucket {
            Bucket {
                size: a.size + b.size,
                aabb: a.aabb.join(&b.aabb),
            }
        }

        // Create six buckets, and six index assignment vectors
        const NUM_BUCKETS: usize = 6;
        let mut buckets = [Bucket::empty(); NUM_BUCKETS];
        let mut bucket_assignments: [Vec<usize>; NUM_BUCKETS] = Default::default();

        // Assign each shape to a bucket
        for idx in &indices {
            let shape = &shapes[*idx];
            let shape_aabb = shape.aabb();
            let shape_center = shape_aabb.center();

            // Get the relative position of the shape centroid [0.0..1.0]
            let bucket_num_relative = (shape_center[split_axis] - centroid_bounds.min[split_axis]) /
                                      split_axis_size;

            // Convert that to the actual `Bucket` number
            let bucket_num = (bucket_num_relative * (NUM_BUCKETS as f32 - 0.01)) as usize;

            // Extend the selected `Bucket` and add the index to the actual bucket
            buckets[bucket_num].add_aabb(&shape_aabb);
            bucket_assignments[bucket_num].push(*idx);
        }

        // Compute the costs for each configuration and
        // select the configuration with the minimal costs
        let mut min_bucket = 0;
        let mut min_cost = f32::INFINITY;
        let mut child_l_aabb = AABB::empty();
        let mut child_r_aabb = AABB::empty();
        for i in 0..(NUM_BUCKETS - 1) {
            let child_l = buckets.iter().take(i + 1).fold(Bucket::empty(), join_bucket);
            let child_r = buckets.iter().skip(i + 1).fold(Bucket::empty(), join_bucket);

            let cost = (child_l.size as f32 * child_l.aabb.surface_area() +
                        child_r.size as f32 * child_r.aabb.surface_area()) /
                       aabb_bounds.surface_area();

            if cost < min_cost {
                min_bucket = i;
                min_cost = cost;
                child_l_aabb = child_l.aabb;
                child_r_aabb = child_r.aabb;
            }
        }

        // Join together all index buckets, and proceed recursively
        let mut child_l_indices = Vec::new();
        for mut indices in bucket_assignments.iter_mut().take(min_bucket + 1) {
            child_l_indices.append(&mut indices);
        }
        let mut child_r_indices = Vec::new();
        for mut indices in bucket_assignments.iter_mut().skip(min_bucket + 1) {
            child_r_indices.append(&mut indices);
        }

        // Construct the actual data structure
        BVHNode::Node {
            child_l_aabb: child_l_aabb,
            child_l: Box::new(BVHNode::build(shapes, child_l_indices)),
            child_r_aabb: child_r_aabb,
            child_r: Box::new(BVHNode::build(shapes, child_r_indices)),
        }
    }

    /// Prints a textual representation of the recursive [`BVH`] structure.
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    fn pretty_print(&self, depth: usize) {
        let padding: String = repeat(" ").take(depth).collect();
        match *self {
            BVHNode::Node { ref child_l, ref child_r, .. } => {
                println!("{}child_l", padding);
                child_l.pretty_print(depth + 1);
                println!("{}child_r", padding);
                child_r.pretty_print(depth + 1);
            }
            BVHNode::Leaf { ref shapes } => {
                println!("{}shapes\t{:?}", padding, shapes);
            }
        }
    }

    /// Traverses the [`BVH`] recursively and insterts shapes which are hit with a
    /// high probability by `ray` into the [`Vec`] `indices`.
    ///
    /// [`BVH`]: struct.BVH.html
    /// [`Vec`]: https://doc.rust-lang.org/std/vec/struct.Vec.html
    ///
    pub fn traverse_recursive(&self, ray: &Ray, indices: &mut Vec<usize>) {
        match *self {
            BVHNode::Node { ref child_l_aabb, ref child_l, ref child_r_aabb, ref child_r } => {
                if ray.intersects_aabb(child_l_aabb) {
                    child_l.traverse_recursive(ray, indices);
                }
                if ray.intersects_aabb(child_r_aabb) {
                    child_r.traverse_recursive(ray, indices);
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

/// The [`BVH`] data structure. Only contains the root node of the [`BVH`] tree.
///
/// [`BVH`]: struct.BVH.html
/// [`build`]: struct.BVH.html#method.build
///
pub struct BVH {
    /// The root node of the [`BVH`].
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    pub root: BVHNode,
}

impl BVH {
    /// Creates a new [`BVH`] from the `shapes` slice.
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    /// # Examples
    ///
    /// ```
    /// use bvh::aabb::{AABB, Bounded};
    /// use bvh::bvh::BVH;
    /// use bvh::nalgebra::{Point3, Vector3};
    ///
    /// # struct Sphere {
    /// #     position: Point3<f32>,
    /// #     radius: f32,
    /// # }
    /// #
    /// # impl Bounded for Sphere {
    /// #     fn aabb(&self) -> AABB {
    /// #         let half_size = Vector3::new(self.radius, self.radius, self.radius);
    /// #         let min = self.position - half_size;
    /// #         let max = self.position + half_size;
    /// #         AABB::with_bounds(min, max)
    /// #     }
    /// # }
    /// #
    /// # fn create_bounded_shapes() -> Vec<Sphere> {
    /// #     let mut spheres = Vec::new();
    /// #     for i in 0..1000u32 {
    /// #         let position = Point3::new(i as f32, i as f32, i as f32);
    /// #         let radius = (i % 10) as f32 + 1.0;
    /// #         spheres.push(Sphere {
    /// #             position: position,
    /// #             radius: radius,
    /// #         });
    /// #     }
    /// #     spheres
    /// # }
    ///
    /// let shapes = create_bounded_shapes();
    /// let bvh = BVH::build(&shapes);
    /// ```
    pub fn build<T: Bounded>(shapes: &[T]) -> BVH {
        let indices = (0..shapes.len()).collect::<Vec<usize>>();
        let root = BVHNode::build(shapes, indices);
        BVH { root: root }
    }

    /// Prints the [`BVH`] in a tree-like visualization.
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    pub fn pretty_print(&self) {
        self.root.pretty_print(0);
    }

    /// Traverses the tree recursively. Returns a subset of `shapes`, in which the [`AABB`]s
    /// of the elements were hit by the `ray`.
    ///
    /// [`AABB`]: ../aabb/struct.AABB.html
    ///
    /// # Examples
    ///
    /// ```
    /// use bvh::aabb::{AABB, Bounded};
    /// use bvh::bvh::BVH;
    /// use bvh::nalgebra::{Point3, Vector3};
    /// use bvh::ray::Ray;
    ///
    /// # struct Sphere {
    /// #     position: Point3<f32>,
    /// #     radius: f32,
    /// # }
    /// #
    /// # impl Bounded for Sphere {
    /// #     fn aabb(&self) -> AABB {
    /// #         let half_size = Vector3::new(self.radius, self.radius, self.radius);
    /// #         let min = self.position - half_size;
    /// #         let max = self.position + half_size;
    /// #         AABB::with_bounds(min, max)
    /// #     }
    /// # }
    /// #
    /// # fn create_bounded_shapes() -> Vec<Sphere> {
    /// #     let mut spheres = Vec::new();
    /// #     for i in 0..1000u32 {
    /// #         let position = Point3::new(i as f32, i as f32, i as f32);
    /// #         let radius = (i % 10) as f32 + 1.0;
    /// #         spheres.push(Sphere {
    /// #             position: position,
    /// #             radius: radius,
    /// #         });
    /// #     }
    /// #     spheres
    /// # }
    ///
    /// let origin = Point3::new(0.0,0.0,0.0);
    /// let direction = Vector3::new(1.0,0.0,0.0);
    /// let ray = Ray::new(origin, direction);
    /// let shapes = create_bounded_shapes();
    /// let bvh = BVH::build(&shapes);
    /// let hit_sphere_aabbs = bvh.traverse_recursive(&ray, &shapes);
    /// ```
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
pub mod tests {
    use std::collections::HashSet;

    use nalgebra::{Point3, Vector3};

    use bvh::BVH;
    use ray::Ray;
    use testbase::{generate_aligned_boxes, UnitBox, create_n_cubes, create_ray};

    /// Creates a `BVH` for a fixed scene structure.
    pub fn build_some_bvh() -> (Vec<UnitBox>, BVH) {
        let boxes = generate_aligned_boxes();
        let bvh = BVH::build(&boxes);
        (boxes, bvh)
    }

    #[test]
    /// Tests whether the building procedure succeeds in not failing.
    fn test_build_bvh() {
        build_some_bvh();
    }

    fn traverse_and_verify(ray_origin: Point3<f32>,
                           ray_direction: Vector3<f32>,
                           all_shapes: &Vec<UnitBox>,
                           bvh: &BVH,
                           expected_shapes: &HashSet<i32>) {
        let ray = Ray::new(ray_origin, ray_direction);
        let hit_shapes = bvh.traverse_recursive(&ray, all_shapes);

        assert_eq!(expected_shapes.len(), hit_shapes.len());
        for shape in hit_shapes {
            assert!(expected_shapes.contains(&shape.id));
        }
    }

    #[test]
    /// Runs some primitive tests for intersections of a ray with a fixed scene given as a BVH.
    fn test_traverse_recursive_bvh() {
        let (all_shapes, bvh) = build_some_bvh();

        {
            // Define a ray which traverses the x-axis from afar.
            let origin = Point3::new(-1000.0, 0.0, 0.0);
            let direction = Vector3::new(1.0, 0.0, 0.0);
            let mut expected_shapes = HashSet::new();

            // It should hit everything.
            for id in -10..11 {
                expected_shapes.insert(id);
            }
            traverse_and_verify(origin, direction, &all_shapes, &bvh, &expected_shapes);
        }

        {
            // Define a ray which traverses the y-axis from afar.
            let origin = Point3::new(0.0, -1000.0, 0.0);
            let direction = Vector3::new(0.0, 1.0, 0.0);

            // It should hit only one box.
            let mut expected_shapes = HashSet::new();
            expected_shapes.insert(0);
            traverse_and_verify(origin, direction, &all_shapes, &bvh, &expected_shapes);
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
            traverse_and_verify(origin, direction, &all_shapes, &bvh, &expected_shapes);
        }
    }

    #[bench]
    /// Benchmark the construction of a BVH with 120,000 triangles.
    fn bench_build_120k_triangles_bvh(b: &mut ::test::Bencher) {
        let triangles = create_n_cubes(10_000);

        b.iter(|| {
            BVH::build(&triangles);
        });
    }

    #[bench]
    /// Benchmark intersecting 120,000 triangles using the recursive BVH.
    fn bench_intersect_120k_triangles_bvh_recursive(b: &mut ::test::Bencher) {
        let triangles = create_n_cubes(10_000);
        let bvh = BVH::build(&triangles);
        let mut seed = 0;

        b.iter(|| {
            let ray = create_ray(&mut seed);

            // Traverse the BVH recursively
            let hits = bvh.traverse_recursive(&ray, &triangles);

            // Traverse the resulting list of positive AABB tests
            for triangle in &hits {
                ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c);
            }
        });
    }
}
