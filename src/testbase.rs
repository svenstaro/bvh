//! Common utilities shared by unit tests.
#![cfg(test)]

use std::collections::HashSet;
use std::f32;

use crate::{Point3, Vector3};
use num::{FromPrimitive, Integer};
use obj::raw::object::Polygon;
use obj::*;
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::aabb::{Bounded, AABB};
use crate::bounding_hierarchy::{BHShape, BoundingHierarchy};
use crate::ray::Ray;

/// A vector represented as a tuple
pub type TupleVec = (f32, f32, f32);

/// Generate a `TupleVec` for [`proptest::strategy::Strategy`] from -10e10 to 10e10
/// A small enough range to prevent most fp32 errors from breaking certain tests
/// Tests which rely on this strategy should probably be rewritten
pub fn tuplevec_small_strategy() -> impl Strategy<Value = TupleVec> {
    (
        -10e10_f32..10e10_f32,
        -10e10_f32..10e10_f32,
        -10e10_f32..10e10_f32,
    )
}

/// Generate a `TupleVec` for [`proptest::strategy::Strategy`] from -10e30 to 10e30
/// A small enough range to prevent `f32::MAX` ranges from breaking certain tests
pub fn tuplevec_large_strategy() -> impl Strategy<Value = TupleVec> {
    (
        -10e30_f32..10e30_f32,
        -10e30_f32..10e30_f32,
        -10e30_f32..10e30_f32,
    )
}

/// Convert a `TupleVec` to a [`Point3`].
pub fn tuple_to_point(tpl: &TupleVec) -> Point3 {
    Point3::new(tpl.0, tpl.1, tpl.2)
}

/// Convert a `TupleVec` to a [`Vector3`].
pub fn tuple_to_vector(tpl: &TupleVec) -> Vector3 {
    Vector3::new(tpl.0, tpl.1, tpl.2)
}

/// Define some `Bounded` structure.
pub struct UnitBox {
    pub id: i32,
    pub pos: Point3,
    node_index: usize,
}

impl UnitBox {
    pub fn new(id: i32, pos: Point3) -> UnitBox {
        UnitBox {
            id,
            pos,
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
pub fn build_some_bh<BH: BoundingHierarchy>() -> (Vec<UnitBox>, BH) {
    let mut boxes = generate_aligned_boxes();
    let bh = BH::build(&mut boxes);
    (boxes, bh)
}

/// Given a ray, a bounding hierarchy, the complete list of shapes in the scene and a list of
/// expected hits, verifies, whether the ray hits only the expected shapes.
fn traverse_and_verify<BH: BoundingHierarchy>(
    ray_origin: Point3,
    ray_direction: Vector3,
    all_shapes: &[UnitBox],
    bh: &BH,
    expected_shapes: &HashSet<i32>,
) {
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
#[derive(Debug)]
pub struct Triangle {
    pub a: Point3,
    pub b: Point3,
    pub c: Point3,
    aabb: AABB,
    node_index: usize,
}

impl Triangle {
    pub fn new(a: Point3, b: Point3, c: Point3) -> Triangle {
        Triangle {
            a,
            b,
            c,
            aabb: AABB::empty().grow(&a).grow(&b).grow(&c),
            node_index: 0,
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

impl<I: FromPrimitive + Integer> FromRawVertex<I> for Triangle {
    fn process(
        vertices: Vec<(f32, f32, f32, f32)>,
        _: Vec<(f32, f32, f32)>,
        _: Vec<(f32, f32, f32)>,
        polygons: Vec<Polygon>,
    ) -> ObjResult<(Vec<Self>, Vec<I>)> {
        // Convert the vertices to `Point3`s.
        let points = vertices
            .into_iter()
            .map(|v| Point3::new(v.0, v.1, v.2))
            .collect::<Vec<_>>();

        // Estimate for the number of triangles, assuming that each polygon is a triangle.
        let mut triangles = Vec::with_capacity(polygons.len());
        {
            let mut push_triangle = |indices: &Vec<usize>| {
                let mut indices_iter = indices.iter();
                let anchor = points[*indices_iter.next().unwrap()];
                let mut second = points[*indices_iter.next().unwrap()];
                for third_index in indices_iter {
                    let third = points[*third_index];
                    triangles.push(Triangle::new(anchor, second, third));
                    second = third;
                }
            };

            // Iterate over the polygons and populate the `Triangle`s vector.
            for polygon in polygons.into_iter() {
                match polygon {
                    Polygon::P(ref vec) => push_triangle(vec),
                    Polygon::PT(ref vec) | Polygon::PN(ref vec) => {
                        push_triangle(&vec.iter().map(|vertex| vertex.0).collect())
                    }
                    Polygon::PTN(ref vec) => {
                        push_triangle(&vec.iter().map(|vertex| vertex.0).collect())
                    }
                }
            }
        }
        Ok((triangles, Vec::new()))
    }
}

/// Creates a unit size cube centered at `pos` and pushes the triangles to `shapes`.
fn push_cube(pos: Point3, shapes: &mut Vec<Triangle>) {
    let top_front_right = pos + Vector3::new(0.5, 0.5, -0.5);
    let top_back_right = pos + Vector3::new(0.5, 0.5, 0.5);
    let top_back_left = pos + Vector3::new(-0.5, 0.5, 0.5);
    let top_front_left = pos + Vector3::new(-0.5, 0.5, -0.5);
    let bottom_front_right = pos + Vector3::new(0.5, -0.5, -0.5);
    let bottom_back_right = pos + Vector3::new(0.5, -0.5, 0.5);
    let bottom_back_left = pos + Vector3::new(-0.5, -0.5, 0.5);
    let bottom_front_left = pos + Vector3::new(-0.5, -0.5, -0.5);

    shapes.push(Triangle::new(
        top_back_right,
        top_front_right,
        top_front_left,
    ));
    shapes.push(Triangle::new(top_front_left, top_back_left, top_back_right));
    shapes.push(Triangle::new(
        bottom_front_left,
        bottom_front_right,
        bottom_back_right,
    ));
    shapes.push(Triangle::new(
        bottom_back_right,
        bottom_back_left,
        bottom_front_left,
    ));
    shapes.push(Triangle::new(
        top_back_left,
        top_front_left,
        bottom_front_left,
    ));
    shapes.push(Triangle::new(
        bottom_front_left,
        bottom_back_left,
        top_back_left,
    ));
    shapes.push(Triangle::new(
        bottom_front_right,
        top_front_right,
        top_back_right,
    ));
    shapes.push(Triangle::new(
        top_back_right,
        bottom_back_right,
        bottom_front_right,
    ));
    shapes.push(Triangle::new(
        top_front_left,
        top_front_right,
        bottom_front_right,
    ));
    shapes.push(Triangle::new(
        bottom_front_right,
        bottom_front_left,
        top_front_left,
    ));
    shapes.push(Triangle::new(
        bottom_back_right,
        top_back_right,
        top_back_left,
    ));
    shapes.push(Triangle::new(
        top_back_left,
        bottom_back_left,
        bottom_back_right,
    ));
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

/// Generates a new `i32` triple. Mutates the seed.
pub fn next_point3_raw(seed: &mut u64) -> (i32, i32, i32) {
    let u = splitmix64(seed);
    let a = ((u >> 32) & 0xFFFFFFFF) as i64 - 0x80000000;
    let b = (u & 0xFFFFFFFF) as i64 - 0x80000000;
    let c = a ^ b.rotate_left(6);
    (a as i32, b as i32, c as i32)
}

/// Generates a new `Point3`, which will lie inside the given `aabb`. Mutates the seed.
pub fn next_point3(seed: &mut u64, aabb: &AABB) -> Point3 {
    let (a, b, c) = next_point3_raw(seed);
    use std::i32;
    let float_vector = Vector3::new(
        (a as f32 / i32::MAX as f32) + 1.0,
        (b as f32 / i32::MAX as f32) + 1.0,
        (c as f32 / i32::MAX as f32) + 1.0,
    ) * 0.5;

    assert!(float_vector.x >= 0.0 && float_vector.x <= 1.0);
    assert!(float_vector.y >= 0.0 && float_vector.y <= 1.0);
    assert!(float_vector.z >= 0.0 && float_vector.z <= 1.0);

    let size = aabb.size();
    let offset = Vector3::new(
        float_vector.x * size.x,
        float_vector.y * size.y,
        float_vector.z * size.z,
    );
    aabb.min + offset
}

/// Returns an `AABB` which defines the default testing space bounds.
pub fn default_bounds() -> AABB {
    AABB::with_bounds(
        Point3::new(-100_000.0, -100_000.0, -100_000.0),
        Point3::new(100_000.0, 100_000.0, 100_000.0),
    )
}

/// Creates `n` deterministic random cubes. Returns the `Vec` of surface `Triangle`s.
pub fn create_n_cubes(n: usize, bounds: &AABB) -> Vec<Triangle> {
    let mut vec = Vec::new();
    let mut seed = 0;
    for _ in 0..n {
        push_cube(next_point3(&mut seed, bounds), &mut vec);
    }
    vec
}

/// Loads the sponza model.
#[cfg(feature = "bench")]
pub fn load_sponza_scene() -> (Vec<Triangle>, AABB) {
    use std::fs::File;
    use std::io::BufReader;

    let file_input =
        BufReader::new(File::open("media/sponza.obj").expect("Failed to open .obj file."));
    let sponza_obj: Obj<Triangle> = load_obj(file_input).expect("Failed to decode .obj file data.");
    let triangles = sponza_obj.vertices;

    let mut bounds = AABB::empty();
    for triangle in &triangles {
        bounds.join_mut(&triangle.aabb());
    }

    (triangles, bounds)
}

/// This functions moves `amount` shapes in the `triangles` array to a new position inside
/// `bounds`. If `max_offset_option` is not `None` then the wrapped value is used as the maximum
/// offset of a shape. This is used to simulate a realistic scene.
/// Returns a `HashSet` of indices of modified triangles.
pub fn randomly_transform_scene(
    triangles: &mut Vec<Triangle>,
    amount: usize,
    bounds: &AABB,
    max_offset_option: Option<f32>,
    seed: &mut u64,
) -> HashSet<usize> {
    let mut indices: Vec<usize> = (0..triangles.len()).collect();
    let mut seed_array = [0u8; 32];
    for i in 0..seed_array.len() {
        let bytes: [u8; 8] = seed.to_be_bytes();
        seed_array[i] = bytes[i % 8];
    }
    let mut rng: StdRng = SeedableRng::from_seed(seed_array);
    indices.shuffle(&mut rng);
    indices.truncate(amount);

    let max_offset = if let Some(value) = max_offset_option {
        value
    } else {
        f32::INFINITY
    };

    for index in &indices {
        let aabb = triangles[*index].aabb();
        let min_move_bound = bounds.min - aabb.min;
        let max_move_bound = bounds.max - aabb.max;
        let movement_bounds = AABB::with_bounds(min_move_bound, max_move_bound);

        let mut random_offset = next_point3(seed, &movement_bounds);
        random_offset.x = max_offset.min((-max_offset).max(random_offset.x));
        random_offset.y = max_offset.min((-max_offset).max(random_offset.y));
        random_offset.z = max_offset.min((-max_offset).max(random_offset.z));

        let triangle = &mut triangles[*index];
        let old_index = triangle.bh_node_index();
        *triangle = Triangle::new(
            triangle.a + random_offset,
            triangle.b + random_offset,
            triangle.c + random_offset,
        );
        triangle.set_bh_node_index(old_index);
    }

    indices.into_iter().collect()
}

/// Creates a `Ray` from the random `seed`. Mutates the `seed`.
/// The Ray origin will be inside the `bounds` and point to some other point inside this
/// `bounds`.
#[cfg(feature = "bench")]
pub fn create_ray(seed: &mut u64, bounds: &AABB) -> Ray {
    let origin = next_point3(seed, bounds);
    let direction = next_point3(seed, bounds);
    Ray::new(origin, direction)
}

/// Benchmark the construction of a `BoundingHierarchy` with `n` triangles.
#[cfg(feature = "bench")]
fn build_n_triangles_bh<T: BoundingHierarchy>(n: usize, b: &mut ::test::Bencher) {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(n, &bounds);
    b.iter(|| {
        T::build(&mut triangles);
    });
}

/// Benchmark the construction of a `BoundingHierarchy` with 1,200 triangles.
#[cfg(feature = "bench")]
pub fn build_1200_triangles_bh<T: BoundingHierarchy>(b: &mut ::test::Bencher) {
    build_n_triangles_bh::<T>(100, b);
}

/// Benchmark the construction of a `BoundingHierarchy` with 12,000 triangles.
#[cfg(feature = "bench")]
pub fn build_12k_triangles_bh<T: BoundingHierarchy>(b: &mut ::test::Bencher) {
    build_n_triangles_bh::<T>(1_000, b);
}

/// Benchmark the construction of a `BoundingHierarchy` with 120,000 triangles.
#[cfg(feature = "bench")]
pub fn build_120k_triangles_bh<T: BoundingHierarchy>(b: &mut ::test::Bencher) {
    build_n_triangles_bh::<T>(10_000, b);
}

/// Benchmark intersecting the `triangles` list without acceleration structures.
#[cfg(feature = "bench")]
pub fn intersect_list(triangles: &[Triangle], bounds: &AABB, b: &mut ::test::Bencher) {
    let mut seed = 0;
    b.iter(|| {
        let ray = create_ray(&mut seed, bounds);

        // Iterate over the list of triangles.
        for triangle in triangles {
            ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c);
        }
    });
}

#[cfg(feature = "bench")]
#[bench]
/// Benchmark intersecting 120,000 triangles directly.
fn bench_intersect_120k_triangles_list(b: &mut ::test::Bencher) {
    let bounds = default_bounds();
    let triangles = create_n_cubes(10_000, &bounds);
    intersect_list(&triangles, &bounds, b);
}

#[cfg(feature = "bench")]
#[bench]
/// Benchmark intersecting Sponza.
fn bench_intersect_sponza_list(b: &mut ::test::Bencher) {
    let (triangles, bounds) = load_sponza_scene();
    intersect_list(&triangles, &bounds, b);
}

/// Benchmark intersecting the `triangles` list with `AABB` checks, but without acceleration
/// structures.
#[cfg(feature = "bench")]
pub fn intersect_list_aabb(triangles: &[Triangle], bounds: &AABB, b: &mut ::test::Bencher) {
    let mut seed = 0;
    b.iter(|| {
        let ray = create_ray(&mut seed, bounds);

        // Iterate over the list of triangles.
        for triangle in triangles {
            // First test whether the ray intersects the AABB of the triangle.
            if ray.intersects_aabb(&triangle.aabb()) {
                ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c);
            }
        }
    });
}

#[cfg(feature = "bench")]
#[bench]
/// Benchmark intersecting 120,000 triangles with preceeding `AABB` tests.
fn bench_intersect_120k_triangles_list_aabb(b: &mut ::test::Bencher) {
    let bounds = default_bounds();
    let triangles = create_n_cubes(10_000, &bounds);
    intersect_list_aabb(&triangles, &bounds, b);
}

#[cfg(feature = "bench")]
#[bench]
/// Benchmark intersecting 120,000 triangles with preceeding `AABB` tests.
fn bench_intersect_sponza_list_aabb(b: &mut ::test::Bencher) {
    let (triangles, bounds) = load_sponza_scene();
    intersect_list_aabb(&triangles, &bounds, b);
}

#[cfg(feature = "bench")]
pub fn intersect_bh<T: BoundingHierarchy>(
    bh: &T,
    triangles: &[Triangle],
    bounds: &AABB,
    b: &mut ::test::Bencher,
) {
    let mut seed = 0;
    b.iter(|| {
        let ray = create_ray(&mut seed, bounds);

        // Traverse the `BoundingHierarchy` recursively.
        let hits = bh.traverse(&ray, triangles);

        // Traverse the resulting list of positive `AABB` tests
        for triangle in &hits {
            ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c);
        }
    });
}

/// Benchmark the traversal of a `BoundingHierarchy` with `n` triangles.
#[cfg(feature = "bench")]
pub fn intersect_n_triangles<T: BoundingHierarchy>(n: usize, b: &mut ::test::Bencher) {
    let bounds = default_bounds();
    let mut triangles = create_n_cubes(n, &bounds);
    let bh = T::build(&mut triangles);
    intersect_bh(&bh, &triangles, &bounds, b)
}

/// Benchmark the traversal of a `BoundingHierarchy` with 1,200 triangles.
#[cfg(feature = "bench")]
pub fn intersect_1200_triangles_bh<T: BoundingHierarchy>(b: &mut ::test::Bencher) {
    intersect_n_triangles::<T>(100, b);
}

/// Benchmark the traversal of a `BoundingHierarchy` with 12,000 triangles.
#[cfg(feature = "bench")]
pub fn intersect_12k_triangles_bh<T: BoundingHierarchy>(b: &mut ::test::Bencher) {
    intersect_n_triangles::<T>(1_000, b);
}

/// Benchmark the traversal of a `BoundingHierarchy` with 120,000 triangles.
#[cfg(feature = "bench")]
pub fn intersect_120k_triangles_bh<T: BoundingHierarchy>(b: &mut ::test::Bencher) {
    intersect_n_triangles::<T>(10_000, b);
}
