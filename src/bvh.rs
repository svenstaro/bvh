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

        /// Left subtree.
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
    use aabb::{AABB, Bounded};
    use bvh::BVH;
    use nalgebra::{Point3, Vector3};
    use std::collections::HashSet;
    use ray::Ray;

    /// Define some Bounded structure.
    pub struct XBox {
        x: i32,
    }

    /// `XBox`'s `AABB`s are unit `AABB`s centered on the given x-position.
    impl Bounded for XBox {
        fn aabb(&self) -> AABB {
            let min = Point3::new(self.x as f32 - 0.5, -0.5, -0.5);
            let max = Point3::new(self.x as f32 + 0.5, 0.5, 0.5);
            AABB::with_bounds(min, max)
        }
    }

    /// Creates a `BVH` for a fixed scene structure.
    pub fn build_some_bvh() -> (Vec<XBox>, BVH) {
        // Create 21 boxes along the x-axis
        let mut shapes = Vec::new();
        for x in -10..11 {
            shapes.push(XBox { x: x });
        }

        let bvh = BVH::build(&shapes);
        (shapes, bvh)
    }

    #[test]
    /// Tests whether the building procedure succeeds in not failing.
    fn test_build_bvh() {
        build_some_bvh();
    }

    #[test]
    /// Runs some primitive tests for intersections of a ray with a fixed scene given as a BVH.
    fn test_traverse_recursive_bvh() {
        let (shapes, bvh) = build_some_bvh();

        // Define a ray which traverses the x-axis from afar
        let position_1 = Point3::new(-1000.0, 0.0, 0.0);
        let direction_1 = Vector3::new(1.0, 0.0, 0.0);
        let ray_1 = Ray::new(position_1, direction_1);

        // It shuold hit all shapes
        let hit_shapes_1 = bvh.traverse_recursive(&ray_1, &shapes);
        assert!(hit_shapes_1.len() == 21);
        let mut xs_1 = HashSet::new();
        for shape in &hit_shapes_1 {
            xs_1.insert(shape.x);
        }
        for x in -10..11 {
            assert!(xs_1.contains(&x));
        }

        // Define a ray which traverses the y-axis from afar
        let position_2 = Point3::new(0.0, -1000.0, 0.0);
        let direction_2 = Vector3::new(0.0, 1.0, 0.0);
        let ray_2 = Ray::new(position_2, direction_2);

        // It should hit only one box
        let hit_shapes_2 = bvh.traverse_recursive(&ray_2, &shapes);
        assert!(hit_shapes_2.len() == 1);
        assert!(hit_shapes_2[0].x == 0);

        // Define a ray which intersects the x-axis diagonally
        let position_3 = Point3::new(6.0, 0.5, 0.0);
        let direction_3 = Vector3::new(-2.0, -1.0, 0.0);
        let ray_3 = Ray::new(position_3, direction_3);

        // It should hit exactly three boxes
        let hit_shapes_3 = bvh.traverse_recursive(&ray_3, &shapes);
        assert!(hit_shapes_3.len() == 3);
        let mut xs_3 = HashSet::new();
        for shape in &hit_shapes_3 {
            xs_3.insert(shape.x);
        }
        assert!(xs_3.contains(&6));
        assert!(xs_3.contains(&5));
        assert!(xs_3.contains(&4));
    }

    /// A triangle struct. Instance of a more complex `Bounded` primitive.
    pub struct Triangle {
        pub a: Point3<f32>,
        pub b: Point3<f32>,
        pub c: Point3<f32>,
        aabb: AABB,
    }

    impl Triangle {
        fn new(a: Point3<f32>, b: Point3<f32>, c: Point3<f32>) -> Triangle {
            Triangle {
                a: a,
                b: b,
                c: c,
                aabb: AABB::empty().grow(&a).grow(&b).grow(&c),
            }
        }
    }

    impl Bounded for Triangle {
        fn aabb(&self) -> AABB {
            self.aabb
        }
    }

    /// Creates a unit size cube centered at `pos` and pushes the triangles to `shapes`
    fn push_cube(pos: Point3<f32>, shapes: &mut Vec<Triangle>) {
        let top_front_right = pos + Vector3::new(0.5, 0.5, -0.5);
        let top_back_right = pos + Vector3::new(0.5, 0.5, 0.5);
        let top_back_left = pos + Vector3::new(-0.5, 0.5, 0.5);
        let top_front_left = pos + Vector3::new(-0.5, 0.5, -0.5);
        let bottom_front_right = pos + Vector3::new(0.5, -0.5, -0.5);
        let bottom_back_right = pos + Vector3::new(0.5, -0.5, 0.5);
        let bottom_back_left = pos + Vector3::new(-0.5, -0.5, 0.5);
        let bottom_front_left = pos + Vector3::new(-0.5, -0.5, -0.5);

        shapes.push(Triangle::new(top_back_right, top_front_right, top_front_left));
        shapes.push(Triangle::new(top_front_left, top_back_left, top_back_right));
        shapes.push(Triangle::new(bottom_front_left, bottom_front_right, bottom_back_right));
        shapes.push(Triangle::new(bottom_back_right, bottom_back_left, bottom_front_left));
        shapes.push(Triangle::new(top_back_left, top_front_left, bottom_front_left));
        shapes.push(Triangle::new(bottom_front_left, bottom_back_left, top_back_left));
        shapes.push(Triangle::new(bottom_front_right, top_front_right, top_back_right));
        shapes.push(Triangle::new(top_back_right, bottom_back_right, bottom_front_right));
        shapes.push(Triangle::new(top_front_left, top_front_right, bottom_front_right));
        shapes.push(Triangle::new(bottom_front_right, bottom_front_left, top_front_left));
        shapes.push(Triangle::new(bottom_back_right, top_back_right, top_back_left));
        shapes.push(Triangle::new(top_back_left, bottom_back_left, bottom_back_right));
    }

    /// Implementation of splitmix64.
    /// For reference see: http://xoroshiro.di.unimi.it/splitmix64.c
    fn splitmix64(x: &mut u64) -> u64 {
        *x = x.wrapping_add(0x9E3779B97F4A7C15u64);
        let mut z = *x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9u64);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EBu64);
        z ^ (z >> 31)
    }

    fn next_point3(seed: &mut u64) -> Point3<f32> {
        let u = splitmix64(seed);
        let a = (u >> 48 & 0xFFFFFFFF) as i32 - 0xFFFF;
        let b = (u >> 48 & 0xFFFFFFFF) as i32 - 0xFFFF;
        let c = a ^ b.rotate_left(6);
        Point3::new(a as f32, b as f32, c as f32)
    }

    pub fn create_n_cubes(n: u64) -> Vec<Triangle> {
        let mut vec = Vec::new();
        let mut seed = 0;

        for _ in 0..n {
            push_cube(next_point3(&mut seed), &mut vec);
        }
        vec
    }

    pub fn create_ray(seed: &mut u64) -> Ray {
        let origin = next_point3(seed);
        let direction = next_point3(seed).to_vector();
        Ray::new(origin, direction)
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
    /// Benchmark intersecting 120,000 triangles directly.
    fn bench_intersect_120k_triangles_list(b: &mut ::test::Bencher) {
        let triangles = create_n_cubes(10_000);
        let mut seed = 0;

        b.iter(|| {
            let ray = create_ray(&mut seed);

            // Iterate over the list of triangles
            for triangle in &triangles {
                ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c);
            }
        });
    }

    #[bench]
    /// Benchmark intersecting 120,000 triangles with preceeding `AABB` tests.
    fn bench_intersect_120k_triangles_list_aabb(b: &mut ::test::Bencher) {
        let triangles = create_n_cubes(10_000);
        let mut seed = 0;

        b.iter(|| {
            let ray = create_ray(&mut seed);

            // Iterate over the list of triangles
            for triangle in &triangles {
                // First test whether the ray intersects the AABB of the triangle
                if ray.intersects_aabb(&triangle.aabb()) {
                    ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c);
                }
            }
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
