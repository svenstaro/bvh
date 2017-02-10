//! Common utilities shared by unit tests.
#![cfg(test)]

use std::collections::HashSet;

use nalgebra::{Point3, Vector3};

use aabb::{AABB, Bounded};
use bounding_hierarchy::{BoundingHierarchy, BHShape};
use ray::Ray;

/// A vector represented as a tuple
pub type TupleVec = (f32, f32, f32);

/// Convert a `TupleVec` to a `nalgebra` point.
pub fn tuple_to_point(tpl: &TupleVec) -> Point3<f32> {
    Point3::new(tpl.0, tpl.1, tpl.2)
}

/// Convert a `TupleVec` to a `nalgebra` vector.
pub fn tuple_to_vector(tpl: &TupleVec) -> Vector3<f32> {
    Vector3::new(tpl.0, tpl.1, tpl.2)
}

/// Define some `Bounded` structure.
pub struct UnitBox {
    pub id: i32,
    pub pos: Point3<f32>,
    node_index: usize,
    // If the XBox's node in the BVH gets moved, we panic if this is true.
    // For testing bvh optimization.
    pub has_good_bh_position: bool,
}

impl UnitBox {
    pub fn new(id: i32, pos: Point3<f32>) -> UnitBox {
        UnitBox {
            id: id,
            pos: pos,
            node_index: 0,
            has_good_bh_position: false,
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
        if self.has_good_bh_position {
            panic!("An UnitBox's BVHNode has been moved even though it shouldn't have.");
        } else {
            self.has_good_bh_position = true;
        }
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
pub fn build_some_bh<BH: BoundingHierarchy>() -> (Vec<UnitBox>, BH) {
    let mut boxes = generate_aligned_boxes();
    let bh = BH::build(&mut boxes);
    (boxes, bh)
}

/// Given a ray, a bounding hierarchy, the complete list of shapes in the scene and a list of
/// expected hits, verifies, whether the ray hits only the expected shapes.
fn traverse_and_verify<BH: BoundingHierarchy>(ray_origin: Point3<f32>,
                                              ray_direction: Vector3<f32>,
                                              all_shapes: &Vec<UnitBox>,
                                              bh: &BH,
                                              expected_shapes: &HashSet<i32>) {
    let ray = Ray::new(ray_origin, ray_direction);
    let hit_shapes = bh.traverse(&ray, all_shapes);

    assert_eq!(expected_shapes.len(), hit_shapes.len());
    for shape in hit_shapes {
        assert!(expected_shapes.contains(&shape.id));
    }
}

/// Perform some fixed intersection tests on BH structures.
pub fn traverse_some_bh<BH: BoundingHierarchy>() {
    let (all_shapes, bh) = build_some_bh::<BH>();

    {
        // Define a ray which traverses the x-axis from afar.
        let origin = Point3::new(-1000.0, 0.0, 0.0);
        let direction = Vector3::new(1.0, 0.0, 0.0);
        let mut expected_shapes = HashSet::new();

        // It should hit everything.
        for id in -10..11 {
            expected_shapes.insert(id);
        }
        traverse_and_verify(origin, direction, &all_shapes, &bh, &expected_shapes);
    }

    {
        // Define a ray which traverses the y-axis from afar.
        let origin = Point3::new(0.0, -1000.0, 0.0);
        let direction = Vector3::new(0.0, 1.0, 0.0);

        // It should hit only one box.
        let mut expected_shapes = HashSet::new();
        expected_shapes.insert(0);
        traverse_and_verify(origin, direction, &all_shapes, &bh, &expected_shapes);
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
        traverse_and_verify(origin, direction, &all_shapes, &bh, &expected_shapes);
    }
}

/// A triangle struct. Instance of a more complex `Bounded` primitive.
pub struct Triangle {
    pub a: Point3<f32>,
    pub b: Point3<f32>,
    pub c: Point3<f32>,
    aabb: AABB,
    node_index: usize,
    has_good_bh_position: bool,
}

impl Triangle {
    pub fn new(a: Point3<f32>, b: Point3<f32>, c: Point3<f32>) -> Triangle {
        Triangle {
            a: a,
            b: b,
            c: c,
            aabb: AABB::empty().grow(&a).grow(&b).grow(&c),
            node_index: 0,
            has_good_bh_position: false,
        }
    }
}

impl Bounded for Triangle {
    fn aabb(&self) -> AABB {
        self.aabb
    }
}

impl BHShape for Triangle {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

/// Creates a unit size cube centered at `pos` and pushes the triangles to `shapes`.
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

/// Generates a new Point3, mutates the seed.
pub fn next_point3(seed: &mut u64) -> Point3<f32> {
    let u = splitmix64(seed);
    let a = ((u >> 32) & 0xFFFFFFFF) as i64 - 0x80000000;
    let b = (u & 0xFFFFFFFF) as i64 - 0x80000000;
    let c = a ^ b.rotate_left(6);
    Point3::new(a as f32, b as f32, c as f32) / 100_000.0
}

/// Creates `n` deterministic random cubes. Returns the `Vec` of surface `Triangle`s.
pub fn create_n_cubes(n: u64) -> Vec<Triangle> {
    let mut vec = Vec::new();
    let mut seed = 0;

    for _ in 0..n {
        push_cube(next_point3(&mut seed), &mut vec);
    }
    vec
}

/// Creates a `Ray` from the random `seed`. Mutates the `seed`.
pub fn create_ray(seed: &mut u64) -> Ray {
    let origin = next_point3(seed);
    let direction = next_point3(seed).to_vector();
    Ray::new(origin, direction)
}

/// Benchmark the construction of a `BoundingHierarchy` with 120,000 triangles.
fn build_n_triangles_bh<T: BoundingHierarchy>(n: u64, b: &mut ::test::Bencher) {
    let mut triangles = create_n_cubes(n);
    b.iter(|| {
        T::build(&mut triangles);
    });
}

/// Benchmark the construction of a `BoundingHierarchy` with 1,200 triangles.
pub fn build_1200_triangles_bh<T: BoundingHierarchy>(b: &mut ::test::Bencher) {
    build_n_triangles_bh::<T>(100, b);
}

/// Benchmark the construction of a `BoundingHierarchy` with 12,000 triangles.
pub fn build_12k_triangles_bh<T: BoundingHierarchy>(b: &mut ::test::Bencher) {
    build_n_triangles_bh::<T>(1_000, b);
}

/// Benchmark the construction of a `BoundingHierarchy` with 120,000 triangles.
pub fn build_120k_triangles_bh<T: BoundingHierarchy>(b: &mut ::test::Bencher) {
    build_n_triangles_bh::<T>(10_000, b);
}

#[bench]
/// Benchmark intersecting 120,000 triangles directly.
fn bench_intersect_120k_triangles_list(b: &mut ::test::Bencher) {
    let triangles = create_n_cubes(10_000);
    let mut seed = 0;

    b.iter(|| {
        let ray = create_ray(&mut seed);

        // Iterate over the list of triangles.
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

        // Iterate over the list of triangles.
        for triangle in &triangles {
            // First test whether the ray intersects the AABB of the triangle.
            if ray.intersects_aabb(&triangle.aabb()) {
                ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c);
            }
        }
    });
}

/// Benchmark the traversal of a `BoundingHierarchy` with `n` triangles.
pub fn intersect_n_triangles<T: BoundingHierarchy>(n: u64, b: &mut ::test::Bencher) {
    let mut triangles = create_n_cubes(n);
    let structure = T::build(&mut triangles);
    let mut seed = 0;

    b.iter(|| {
        let ray = create_ray(&mut seed);

        // Traverse the `BoundingHierarchy` recursively.
        let hits = structure.traverse(&ray, &triangles);

        // Traverse the resulting list of positive AABB tests
        for triangle in &hits {
            ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c);
        }
    });
}

/// Benchmark the traversal of a `BoundingHierarchy` with 1,200 triangles.
pub fn intersect_1200_triangles_bh<T: BoundingHierarchy>(b: &mut ::test::Bencher) {
    intersect_n_triangles::<T>(100, b);
}

/// Benchmark the traversal of a `BoundingHierarchy` with 12,000 triangles.
pub fn intersect_12k_triangles_bh<T: BoundingHierarchy>(b: &mut ::test::Bencher) {
    intersect_n_triangles::<T>(1_000, b);
}

/// Benchmark the traversal of a `BoundingHierarchy` with 120,000 triangles.
pub fn intersect_120k_triangles_bh<T: BoundingHierarchy>(b: &mut ::test::Bencher) {
    intersect_n_triangles::<T>(10_000, b);
}
