use aabb::{AABB, Bounded};
use ray::Ray;
use std::boxed::Box;
use std::f32;
use std::iter::repeat;
// TODO kommentar zu wieso keine root aabb
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

pub struct FlatNode {
    aabb: AABB,
    entry_index: u32,
    exit_index: u32,
    shape_index: u32,
}

const MAX_UINT32: u32 = 0xFFFFFFFF;

pub fn print_flat_tree(flat_nodes: &[FlatNode]) {
    for (i, node) in flat_nodes.iter().enumerate() {
        println!("{}\tentry {}\texit {}\tshape {}",
                 i,
                 node.entry_index,
                 node.exit_index,
                 node.shape_index);
    }
}

pub fn traverse_flat_bvh<'a, T: Bounded>(ray: &Ray,
                                         flat_nodes: &[FlatNode],
                                         shapes: &'a [T])
                                         -> Vec<&'a T> {
    let mut hit_shapes = Vec::new();
    let mut index = 0;
    let max_length = flat_nodes.len();
    while index < max_length {
        let node = &flat_nodes[index];
        if node.entry_index == MAX_UINT32 {
            let shape_index = node.shape_index;
            let actual_shape = &shapes[shape_index as usize];
            let actual_aabb = actual_shape.aabb();
            if ray.intersects_aabb(&actual_aabb) {
                hit_shapes.push(actual_shape);
            }
            index = node.exit_index as usize;
        } else if ray.intersects_aabb(&node.aabb) {
            index = node.entry_index as usize;
        } else {
            index = node.exit_index as usize;
        }
    }
    hit_shapes
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
            BVHNode::Node { ref init, ref tail, .. } => {
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

    pub fn flatten_tree(&self, vec: &mut Vec<FlatNode>, next_free: usize) -> usize {
        match *self {
            BVHNode::Node { ref init_aabb, ref init, ref tail_aabb, ref tail } => {

                let init_node = FlatNode {
                    aabb: *init_aabb,
                    entry_index: (next_free + 1) as u32,
                    exit_index: 0,
                    shape_index: MAX_UINT32,
                };
                vec.push(init_node);

                let index_after_init = init.flatten_tree(vec, next_free + 1);
                vec[next_free as usize].exit_index = index_after_init as u32;

                let exit_node = FlatNode {
                    aabb: *tail_aabb,
                    entry_index: (index_after_init + 1) as u32,
                    exit_index: 0,
                    shape_index: MAX_UINT32,
                };
                vec.push(exit_node);

                let index_after_tail = tail.flatten_tree(vec, index_after_init + 1);
                vec[index_after_init as usize].exit_index = index_after_tail as u32;

                index_after_tail
            }
            BVHNode::Leaf { ref shapes } => {
                let mut next_shape = next_free;
                for shape_index in shapes {
                    next_shape += 1;
                    let leaf_node = FlatNode {
                        aabb: AABB::empty(),
                        entry_index: MAX_UINT32,
                        exit_index: next_shape as u32,
                        shape_index: *shape_index as u32,
                    };
                    vec.push(leaf_node);
                }

                next_shape
            }
        }
    }
}

/// The BVH data structure. Only contains the BVH structure and indices to
/// the slice of shapes given in the `new` function.
pub struct BVH {
    /// The root node of the BVH
    root: BVHNode,
}

impl BVH {
    /// Creates a new BVH from the slice of shapes.
    pub fn new<T: Bounded>(shapes: &[T]) -> BVH {
        let indices = (0..shapes.len()).collect::<Vec<usize>>();
        let root = BVHNode::new(shapes, indices);
        BVH { root: root }
    }

    /// Prints the BVH in a tree-like visualization.
    pub fn print(&self) {
        self.root.print(0);
    }

    /// Traverses the tree recursively. Returns an array of all shapes which were hit.
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

    /// Flattens the BVH so that it can be traversed iteratively.
    pub fn flatten_tree(&self) -> Vec<FlatNode> {
        let mut vec = Vec::new();
        self.root.flatten_tree(&mut vec, 0);
        vec
    }
}

#[cfg(test)]
mod tests {
    use aabb::{AABB, Bounded};
    use bvh::{BVH, traverse_flat_bvh, print_flat_tree};
    use nalgebra::{Point3, Vector3};
    use std::collections::HashSet;
    use ray::Ray;

    /// Define some Bounded structure.
    struct XBox {
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
    fn build_some_bvh() -> (Vec<XBox>, BVH) {
        // Create 21 boxes along the x-axis
        let mut shapes = Vec::new();
        for x in -10..11 {
            shapes.push(XBox { x: x });
        }

        let bvh = BVH::new(&shapes);
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

    #[test]
    /// Tests whether the `flatten_tree` procedure succeeds in not failing.
    fn test_flatten_bvh() {
        let (_, bvh) = build_some_bvh();
        bvh.flatten_tree();
    }

    #[test]
    /// Runs some primitive tests for intersections of a ray with
    /// a fixed scene given as a flat BVH.
    fn test_traverse_flat_bvh() {
        let (shapes, bvh) = build_some_bvh();
        let flat_bvh = bvh.flatten_tree();
        print_flat_tree(&flat_bvh);

        // Define a ray which traverses the x-axis from afar
        let position_1 = Point3::new(-1000.0, 0.0, 0.0);
        let direction_1 = Vector3::new(1.0, 0.0, 0.0);
        let ray_1 = Ray::new(position_1, direction_1);

        // It shuold hit all shapes
        let hit_shapes_1 = traverse_flat_bvh(&ray_1, &flat_bvh, &shapes);
        assert!(hit_shapes_1.len() == 21);
        let mut xs_1 = HashSet::new();
        for shape in &hit_shapes_1 {
            xs_1.insert(shape.x);
        }
        for x in -10..11 {
            assert!(xs_1.contains(&x));
        }

        // Define a ray which traverses the y-axis from afar
        let position_2 = Point3::new(1.0, -1000.0, 0.0);
        let direction_2 = Vector3::new(0.0, 1.0, 0.0);
        let ray_2 = Ray::new(position_2, direction_2);

        // It should hit only one box
        let hit_shapes_2 = traverse_flat_bvh(&ray_2, &flat_bvh, &shapes);
        assert!(hit_shapes_2.len() == 1);
        assert!(hit_shapes_2[0].x == 1);

        // Define a ray which intersects the x-axis diagonally
        let position_3 = Point3::new(6.0, 0.5, 0.0);
        let direction_3 = Vector3::new(-2.0, -1.0, 0.0);
        let ray_3 = Ray::new(position_3, direction_3);

        // It should hit exactly three boxes
        let hit_shapes_3 = traverse_flat_bvh(&ray_3, &flat_bvh, &shapes);
        assert!(hit_shapes_3.len() == 3);
        let mut xs_3 = HashSet::new();
        for shape in &hit_shapes_3 {
            xs_3.insert(shape.x);
        }
        assert!(xs_3.contains(&6));
        assert!(xs_3.contains(&5));
        assert!(xs_3.contains(&4));
    }

    struct Triangle {
        a: Point3<f32>,
        b: Point3<f32>,
        c: Point3<f32>,
    }

    impl Triangle {
        fn new(a: Point3<f32>, b: Point3<f32>, c: Point3<f32>) -> Triangle {
            Triangle { a: a, b: b, c: c }
        }
    }

    impl Bounded for Triangle {
        fn aabb(&self) -> AABB {
            let mut min = self.a;
            let mut max = self.a;
            min.x = min.x.min(self.b.x).min(self.c.x);
            min.y = min.y.min(self.b.y).min(self.c.y);
            min.z = min.z.min(self.b.z).min(self.c.z);
            max.x = max.x.max(self.b.x).max(self.c.x);
            max.y = max.y.max(self.b.y).max(self.c.y);
            max.z = max.z.max(self.b.z).max(self.c.z);
            AABB::with_bounds(min, max)
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
    unsafe fn splitmix64(x: &mut u64) -> u64 {
        *x += 0x9E3779B97F4A7C15u64;
        let mut z = *x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9u64;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBu64;
        z ^ (z >> 31)
    }

    unsafe fn u64_to_f32(a: u64) -> (f32, f32) {
        let adr_u64 = (&a as *const u64) as usize;
        (*(adr_u64 as *const f32), *((adr_u64 + 1) as *const f32))
    }

    fn next_point3(seed: &mut u64) -> Point3<f32> {
        unsafe {
            let (a, b) = u64_to_f32(splitmix64(seed));
            let (c, _) = u64_to_f32(splitmix64(seed));
            Point3::new(a, b, c)
        }
    }

    fn create_100k_cubes() -> Vec<Triangle> {
        let mut vec = Vec::new();
        let mut seed = 0;

        for _ in 0..100_000 {
            push_cube(next_point3(&mut seed), &mut vec);
        }
        vec
    }

    #[bench]
    /// Benchmark for the branchless intersection algorithm.
    fn bench_build_1200k_triangles_bvh(b: &mut ::test::Bencher) {
        let cubes = create_100k_cubes();

        b.iter(|| {
            BVH::new(&cubes);
        });
    }
}
